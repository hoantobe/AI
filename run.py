import random
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
from pathlib import Path
import uuid
import time
import logging
from threading import Thread
from app.gemini_translate import GeminiTranslator
import pysrt
from dotenv import load_dotenv
import requests
from datetime import datetime, timedelta
import threading
import re

from app.services import translator

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Khởi tạo Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit

# Tạo thư mục uploads nếu chưa tồn tại
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load biến môi trường
load_dotenv()

class TaskManager:
    def __init__(self, timeout=3600):
        self.tasks = {}
        self.timestamps = {}
        self.timeout = timeout
        self.lock = threading.Lock()
        self.requests_per_minute = 15
        self.request_times = []

    def create_task(self, config):
        """Tạo task mới với ID unique"""
        with self.lock:
            task_id = str(uuid.uuid4())
            self.tasks[task_id] = {
                'status': 'processing',
                'progress': 0,
                'created_at': time.time(),
                'last_activity': time.time(),
                **config
            }
            self.timestamps[task_id] = datetime.now()
            return task_id

    def get_task(self, task_id):
        """Lấy thông tin task với validation"""
        with self.lock:
            if task_id not in self.tasks:
                return None
                
            task = self.tasks[task_id]
            # Cập nhật thời gian hoạt động
            task['last_activity'] = time.time()
            self.timestamps[task_id] = datetime.now()
            return task

    def update_task(self, task_id, updates):
        """Cập nhật thông tin task"""
        with self.lock:
            if task_id in self.tasks:
                self.tasks[task_id].update(updates)
                self.tasks[task_id]['last_activity'] = time.time()
                self.timestamps[task_id] = datetime.now()

    def remove_task(self, task_id):
        """Xóa task"""
        with self.lock:
            if task_id in self.tasks:
                del self.tasks[task_id]
            if task_id in self.timestamps:
                del self.timestamps[task_id]

    def cleanup_old_tasks(self):
        """Dọn dẹp các task cũ"""
        with self.lock:
            current_time = time.time()
            expired = []
            for task_id, timestamp in self.timestamps.items():
                if (datetime.now() - timestamp).total_seconds() > self.timeout:
                    expired.append(task_id)
            
            for task_id in expired:
                self.remove_task(task_id)

    def check_rate_limit(self):
        """Kiểm tra rate limit"""
        now = time.time()
        self.request_times = [t for t in self.request_times if now - t < 60]
        return len(self.request_times) < self.requests_per_minute

    def add_request(self):
        """Thêm request mới"""
        self.request_times.append(time.time())

# Khởi tạo task manager global
task_manager = TaskManager(timeout=3600)
@app.route('/')
def home():
    """Trang chủ"""
    return render_template('home.html')


@app.route('/translate')
def dich():
    """Trang chủ"""
    return render_template('index.html')



@app.route('/translate', methods=['POST'])
def translate():
    """API xử lý yêu cầu dịch file"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Không tìm thấy file'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Chưa chọn file'}), 400

        # Lưu file với tên an toàn
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Tạo tên file output
        output_filename = f"{Path(filename).stem}_translated.srt"
        output_filepath = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)

        source_lang = request.form.get('source_lang', 'zh')
        target_lang = request.form.get('target_lang', 'vi')
        # Lấy các tùy chọn từ form
        style_config = {
            
            "tone": request.form.get('tone', 'standard'),
            "audience": request.form.get('audience', 'general'),
            "creativity": float(request.form.get('creativity', 0.1)),
            "source_lang": source_lang,
            "target_lang": target_lang,
            "use_cache": request.form.get('use_cache', 'false') == 'true',
            "split_config": {
                "enabled": request.form.get('auto_split', 'false') == 'true',
                "max_length": int(request.form.get('max_length', 45))
            }
        }

        # Tạo task mới
        task_id = task_manager.create_task({
            'filepath': filepath,
            'output_filepath': output_filepath,
            'style_config': style_config
        })

        # Khởi chạy task trong thread riêng
        translator = GeminiTranslator(task_id=task_id, style_config=style_config)
        thread = Thread(
            target=translate_async,
            args=(task_id, translator, filepath, output_filepath)
        )
        thread.daemon = True
        thread.start()

        return jsonify({'task_id': task_id})

    except Exception as e:
        logger.error(f"Lỗi: {str(e)}")
        return jsonify({'error': str(e)}), 500

def translate_async(task_id, translator, input_file, output_file):
    """Hàm xử lý dịch bất đồng bộ"""
    try:
        logger.info(f"Bắt đầu dịch task {task_id}")
        task = task_manager.get_task(task_id)
        if not task:
            raise ValueError(f"Task {task_id} không tồn tại")

        # Đọc file với encoding phù hợp
        subs = pysrt.open(input_file, encoding='utf-8')
        total_subs = len(subs)
        
        # Khởi tạo translated_subs ngay từ đầu
        translated_subs = pysrt.SubRipFile()

        # Cập nhật thông tin task ban đầu
        task_manager.update_task(task_id, {
            'total_subs': total_subs,
            'current_sub': 0,
            'progress': 0,
            'status': 'processing'
        })
        logger.info(f"Tổng số phụ đề cần dịch: {total_subs}")

        # Chia nhỏ thành các batch nhỏ hơn
        batch_size = 20  # Giảm batch size
        batches = [subs[i:i + batch_size] for i in range(0, total_subs, batch_size)]
        
        for batch_idx, batch in enumerate(batches):
            try:
                # Log thông tin batch
                logger.info(f"Đang xử lý batch {batch_idx + 1}/{len(batches)}")
                
                # Dịch batch hiện tại
                texts = [sub.text.strip() for sub in batch]
                translations = translator.translate_batch(texts)
                
                if not translations:
                    logger.error(f"Không nhận được kết quả dịch cho batch {batch_idx + 1}")
                    continue

                # Cập nhật subtitles
                for sub, trans in zip(batch, translations):
                    new_sub = pysrt.SubRipItem(
                        index=len(translated_subs) + 1,
                        start=sub.start,
                        end=sub.end,
                        text=trans
                    )
                    translated_subs.append(new_sub)

                # Cập nhật tiến độ
                progress = min(100, int((batch_idx + 1) * 100 / len(batches)))
                task_manager.update_task(task_id, {
                    'progress': progress,
                    'current_sub': (batch_idx + 1) * batch_size,
                    'status': 'processing',
                    'last_translation': translations[-1] if translations else None
                })
                logger.info(f"Tiến độ: {progress}%")

                # Lưu kết quả tạm thời
                if batch_idx % 2 == 0:
                    temp_output = f"{output_file}.tmp"
                    translated_subs.save(temp_output, encoding='utf-8')

            except Exception as batch_error:
                logger.error(f"Lỗi batch {batch_idx}: {str(batch_error)}")
                continue

            # Giảm delay giữa các batch
            time.sleep(0.5)

        # Lưu kết quả cuối cùng
        translated_subs.save(output_file, encoding='utf-8')
        logger.info(f"Hoàn thành dịch task {task_id}")
        
        task_manager.update_task(task_id, {
            'status': 'completed',
            'progress': 100,
            'completion_time': time.time()
        })

    except Exception as e:
        logger.error(f"Lỗi task {task_id}: {str(e)}")
        task_manager.update_task(task_id, {
            'status': 'failed',
            'error': str(e)
        })

@app.route('/progress/<task_id>')
def get_progress(task_id):
    """API kiểm tra tiến độ của task"""
    try:
        task = task_manager.get_task(task_id)
        
        if not task:
            logger.warning(f"Task {task_id} không tồn tại")
            return jsonify({
                'status': 'expired',
                'error': 'Task đã hết hạn hoặc không tồn tại'
            }), 404

        # Thêm thông tin chi tiết
        response = {
            'status': task.get('status', 'unknown'),
            'progress': task.get('progress', 0),
            'current_sub': task.get('current_sub', 0),
            'total_subs': task.get('total_subs', 0),
            'translated_text': task.get('translated_text', ''),
            'last_translation': task.get('last_translation', ''),
            'summary': task.get('summary', ''),  # Thêm trường summary
            'error': task.get('error', None)
        }
        
        return jsonify(response)

    except Exception as e:
        logger.error(f"Lỗi khi lấy tiến độ task {task_id}: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

def update_task_progress(task_id, progress):
    """Cập nhật tiến độ task"""
    task_manager.update_task(task_id, {
        'progress': progress,
        'last_progress_update': time.time()
    })

@app.route('/download/<task_id>')
def download_file(task_id):
    """API tải file kết quả"""
    task = task_manager.get_task(task_id)
    if task and task['status'] == 'completed':
        return send_file(
            task['output_filepath'],
            as_attachment=True,
            download_name=os.path.basename(task['output_filepath'])
        )
    return jsonify({'error': 'File không tồn tại'}), 404

@app.route('/toggle_pause', methods=['POST'])
def toggle_pause():
    """API tạm dừng/tiếp tục task"""
    task_id = request.form.get('task_id')
    is_paused = request.form.get('is_paused') == 'true'
    task_manager.update_task(task_id, {'is_paused': is_paused})
    return jsonify({'success': True})

@app.route('/text-to-srt')  
def text_to_srt_page():
    """Trang chuyển văn bản thành phụ đề"""
    return render_template('text_to_srt.html')

@app.route('/text-to-srt/convert', methods=['POST'])
def convert_text_to_srt():
    """API xử lý chuyển văn bản thành phụ đề"""
    try:
        text = request.form.get('text')
        if not text:
            return jsonify({'error': 'Văn bản không được để trống'}), 400

        # Tạo file output
        output_filename = f"subtitle_{int(time.time())}.srt"
        output_filepath = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)

        # Tạo task mới
        task_id = task_manager.create_task({
            'output_filepath': output_filepath,
            'text_length': len(text),
            'status': 'processing'
        })

        # Xử lý trong thread riêng
        thread = Thread(
            target=process_text_to_srt,
            args=(task_id, text, output_filepath)
        )
        thread.daemon = True
        thread.start()

        return jsonify({'task_id': task_id})

    except Exception as e:
        logger.error(f"Lỗi: {str(e)}")
        return jsonify({'error': str(e)}), 500

def process_text_to_srt(task_id, text, output_file):
    """Hàm xử lý chuyển văn bản thành phụ đề"""
    try:
        subs = pysrt.SubRipFile()
        current_index = 1
        
        # Chia văn bản thành các đoạn 500 ký tự
        segments = split_text_to_segments(text, 500)
        total_segments = len(segments)
        
        # Tạo phụ đề cho mỗi đoạn
        for i, segment in enumerate(segments):
            # Tính thời gian (30 giây mỗi phụ đề)
            start_time = i * 30  # Thay đổi từ 3s thành 30s
            end_time = (i + 1) * 30
            
            sub = pysrt.SubRipItem(
                index=current_index,
                start=pysrt.SubRipTime(seconds=start_time),
                end=pysrt.SubRipTime(seconds=end_time),
                text=segment
            )
            subs.append(sub)
            current_index += 1
            
            # Cập nhật tiến độ
            progress = min(100, int((i + 1) * 100 / total_segments))
            task_manager.update_task(task_id, {
                'progress': progress,
                'current_segment': i + 1,
                'total_segments': total_segments,
                'duration': 30  # Thêm thông tin về thời lượng
            })

        # Lưu file
        subs.save(output_file, encoding='utf-8')
        
        task_manager.update_task(task_id, {
            'status': 'completed',
            'progress': 100,
            'total_duration': total_segments * 30  # Tổng thời lượng
        })
        
    except Exception as e:
        logger.error(f"Lỗi khi xử lý văn bản: {str(e)}")
        task_manager.update_task(task_id, {
            'status': 'failed',
            'error': str(e)
        })

def split_text_to_segments(text, max_length=500):
    """Chia văn bản thành các đoạn có độ dài tối đa"""
    segments = []
    
    # Tách thành các câu
    sentences = re.split(r'([.!?]+[\s\n]+)', text)
    current_segment = []
    current_length = 0
    
    for sentence in sentences:
        if not sentence.strip():
            continue
            
        # Nếu câu dài hơn max_length, chia nhỏ theo dấu phẩy
        if len(sentence) > max_length:
            parts = re.split(r'(,+[\s\n]+)', sentence)
            for part in parts:
                if not part.strip():
                    continue
                    
                if current_length + len(part) <= max_length:
                    current_segment.append(part)
                    current_length += len(part)
                else:
                    if current_segment:
                        segments.append(''.join(current_segment).strip())
                    current_segment = [part]
                    current_length = len(part)
        else:
            if current_length + len(sentence) <= max_length:
                current_segment.append(sentence)
                current_length += len(sentence)
            else:
                if current_segment:
                    segments.append(''.join(current_segment).strip())
                current_segment = [sentence]
                current_length = len(sentence)
    
    # Thêm đoạn cuối cùng
    if current_segment:
        segments.append(''.join(current_segment).strip())
        
    return segments

def cleanup_old_files():
    """Dọn dẹp file tạm và task cũ"""
    while True:
        try:
            current_time = time.time()
            # Xóa file cũ hơn 1 giờ
            for file in Path(app.config['UPLOAD_FOLDER']).glob('*.*'):
                if current_time - file.stat().st_mtime > 3600:
                    try:
                        os.remove(file)
                    except Exception as e:
                        logger.warning(f"Không thể xóa file {file}: {str(e)}")
            
            # Xóa task cũ
            task_manager.cleanup_old_tasks()
                
            time.sleep(300)  # Kiểm tra mỗi 5 phút
            
        except Exception as e:
            logger.error(f"Lỗi khi dọn dẹp: {str(e)}")
            time.sleep(60)

class APIKeyManager:
    def __init__(self):
        self.keys = {}
        self.lock = threading.Lock()
        self.load_keys_from_env()

    def load_keys_from_env(self):
        """Load all API keys from .env file"""
        for i in range(1, 10):  # Hỗ trợ tối đa 9 keys
            key = os.getenv(f'GEMINI_API_KEY_{i}')
            if not key:
                break
            self.add_key(key)
        logger.info(f"Loaded {len(self.keys)} API keys")

    def add_key(self, api_key):
        """Add new API key with rate limit tracking"""
        with self.lock:
            if api_key not in self.keys:
                self.keys[api_key] = {
                    'requests': 0,
                    'last_reset': datetime.now(),
                    'is_active': True,
                    'errors': 0
                }

    def get_available_key(self):
        """Get an available API key"""
        with self.lock:
            available_keys = [k for k, v in self.keys.items() if v['is_active']]
            if not available_keys:
                logger.error("No available API keys")
                return None
            key = random.choice(available_keys)
            logger.info(f"Using API key: {key[:10]}...")
            return key

    def mark_key_error(self, api_key):
        """Mark key as having error"""
        with self.lock:
            if api_key in self.keys:
                self.keys[api_key]['errors'] += 1
                if self.keys[api_key]['errors'] >= 3:
                    self.keys[api_key]['is_active'] = False
                logger.warning(f"API key marked error (errors: {self.keys[api_key]['errors']})")

    def reset_key(self, api_key):
        """Reset key status"""
        with self.lock:
            if api_key in self.keys:
                self.keys[api_key]['is_active'] = True
                self.keys[api_key]['errors'] = 0
                self.keys[api_key]['requests'] = 0
                self.keys[api_key]['last_reset'] = datetime.now()

def _call_api(self, payload):
    """Call API with key rotation"""
    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        api_key = self.key_manager.get_available_key()
        if not api_key:
            logger.error("Không có API key khả dụng")
            time.sleep(2)
            continue

        try:
            logger.info(f"Gọi API với batch size {len(payload['contents'][0]['parts'][0]['text'].split(chr(10)))}")
            
            self.session.headers.update({
                'x-goog-api-key': api_key,
                'Content-Type': 'application/json'
            })
            
            response = self.session.post(
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent",
                json=payload,
                timeout=30
            )

            if response.status_code == 429:
                logger.warning(f"Rate limit exceeded for key {api_key}")
                self.key_manager.mark_key_error(api_key)
                continue

            response.raise_for_status()
            response_data = response.json()
            
            # Log response để debug
            logger.info(f"API Response: {response_data}")
            
            return response_data

        except Exception as e:
            logger.error(f"API call failed: {str(e)}")
            self.key_manager.mark_key_error(api_key)
            retry_count += 1
            if retry_count == max_retries:
                raise
            time.sleep(2 ** retry_count)

def smart_split_with_ai(text, config, translator):
    """Tách phụ đề thông minh sử dụng AI"""
    try:
        prompt = (
            f"Split this subtitle text into shorter lines:\n\n"
            f"Rules:\n"
            f"- Maximum {config['max_length']} characters per line\n"
            f"- Keep meaning and context\n"
            f"- Do not split in the middle of phrases or sentences\n"
            f"- Ensure each line is a complete thought\n"
            f"- Number each line\n\n"
            f"Text to split:\n{text}\n\n"
            f"Output format:\n"
            f"[1] first line\n"
            f"[2] second line\n"
            f"...\n"
            f"Only return the split lines, no explanation or comments."
        )

        response = translator._call_api({
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "temperature": 0.1,  # Giảm temperature để có kết quả ổn định
                "topK": 1,
                "topP": 1,
                "maxOutputTokens": 1024
            }
        })

        if not response or 'candidates' not in response:
            logger.warning("Không nhận được phản hồi từ AI, trả về text gốc")
            return [text]

        # Parse kết quả
        result_text = response['candidates'][0]['content']['parts'][0]['text']
        lines = []

        # Xử lý từng dòng kết quả
        for line in result_text.split('\n'):
            line = line.strip()
            if not line:
                continue

            # Tìm nội dung sau số thứ tự
            match = re.match(r'(?:\[)?(\d+)(?:\])?\s*[:：]?\s*(.*)', line)
            if match and match.group(2):
                split_text = match.group(2).strip()
                lines.append(split_text)

        # Kiểm tra kết quả
        if not lines:
            logger.warning("Không có dòng nào sau khi tách, trả về text gốc")
            return [text]

        logger.info(f"Đã tách thành {len(lines)} dòng")
        return lines

    except Exception as e:
        logger.error(f"Lỗi khi tách bằng AI: {str(e)}")
        return [text]  # Trả về text gốc nếu có lỗi

@app.route('/split-subtitle', methods=['GET'])
def split_subtitle_page():
    """Trang tách phụ đề"""
    return render_template('split.html')

@app.route('/split-subtitle', methods=['POST'])
def split_subtitle():
    """API xử lý tách phụ đề"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Không tìm thấy file'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Chưa chọn file'}), 400

        # Thêm timestamp vào tên file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = secure_filename(file.filename)
        base_name = Path(filename).stem
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{base_name}_{timestamp}.srt")
        file.save(filepath)

        # Tạo tên file output với timestamp
        output_filename = f"{base_name}_split_{timestamp}.srt"
        output_filepath = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)

        # Lấy các tùy chọn từ form
        split_config = {
            'max_length': int(request.form.get('max_length', 45)),
            'min_duration': float(request.form.get('min_duration', 1.0)),
            'smart_split': request.form.get('smart_split', 'false') == 'true',
            'use_ai': request.form.get('use_ai', 'false') == 'true'
        }

        # Tạo task mới
        task_id = task_manager.create_task({
            'filepath': filepath,
            'output_filepath': output_filepath,
            'split_config': split_config,
            'status': 'processing'
        })

        # Xử lý trong thread riêng
        thread = Thread(
            target=process_split_subtitle,
            args=(task_id, filepath, output_filepath, split_config)
        )
        thread.daemon = True
        thread.start()

        return jsonify({'task_id': task_id})

    except Exception as e:
        logger.error(f"Lỗi khi tách phụ đề: {str(e)}")
        return jsonify({'error': str(e)}), 500

def process_split_subtitle(task_id, input_file, output_file, config):
    """Hàm xử lý tách phụ đề"""
    try:
        # Đọc file input
        subs = pysrt.open(input_file, encoding='utf-8')
        total_subs = len(subs)
        
        # Khởi tạo danh sách phụ đề mới
        new_subs = pysrt.SubRipFile()
        current_index = 1
        
        task_manager.update_task(task_id, {
            'total_subs': total_subs,
            'current_sub': 0,
            'progress': 0
        })

        for i, sub in enumerate(subs):
            text = sub.text.strip()
            if not text:
                continue

            # Sử dụng AI để tách phụ đề nếu được bật
            if config['use_ai']:
                lines = smart_split_with_ai(text, config, GeminiTranslator(task_id=task_id))
            else:
                # Tách phụ đề nếu quá dài
                lines = split_subtitle_text(
                    text, 
                    max_length=config['max_length'],
                    smart_split=config['smart_split']
                )
                
            # Tính thời gian cho mỗi dòng
            duration = sub.end.ordinal - sub.start.ordinal
            time_per_line = duration / len(lines)
            min_duration_ms = int(config['min_duration'] * 1000)

            # Thêm từng dòng đã tách
            for j, line in enumerate(lines):
                start_time = sub.start.ordinal + (j * time_per_line)
                end_time = start_time + max(time_per_line, min_duration_ms)
                
                new_sub = pysrt.SubRipItem(
                    index=current_index,
                    start=pysrt.SubRipTime.from_ordinal(int(start_time)),
                    end=pysrt.SubRipTime.from_ordinal(int(end_time)),
                    text=line
                )
                new_subs.append(new_sub)
                current_index += 1

            # Cập nhật tiến độ
            progress = min(100, int((i + 1) * 100 / total_subs))
            task_manager.update_task(task_id, {
                'progress': progress,
                'current_sub': i + 1
            })

        # Lưu file kết quả
        new_subs.save(output_file, encoding='utf-8')
        
        task_manager.update_task(task_id, {
            'status': 'completed',
            'progress': 100
        })

    except Exception as e:
        logger.error(f"Lỗi khi xử lý tách phụ đề: {str(e)}")
        task_manager.update_task(task_id, {
            'status': 'failed',
            'error': str(e)
        })


@app.route('/translate_text', methods=['POST'])
def translate_text():
    """API xử lý dịch văn bản trực tiếp"""
    try:
        # Lấy văn bản đầu vào
        text = request.form.get('sourceText')
        if not text:
            return jsonify({'error': 'Văn bản không được để trống'}), 400

        logger.info(f"Nhận yêu cầu dịch văn bản: {len(text)} ký tự")

        # Lấy các tùy chọn từ form
        style_config = {
            "tone": request.form.get('tone', 'standard'),
            "audience": request.form.get('audience', 'general'),
            "creativity": float(request.form.get('creativity', 0.1)),
            "source_lang": request.form.get('source_lang', 'zh'),
            "target_lang": request.form.get('target_lang', 'vi')
        }

        # Tạo task mới
        task_id = task_manager.create_task({
            'text': text,
            'style_config': style_config,
            'status': 'processing',
            'translated_text': '',  # Thêm trường này
            'total_lines': 1,      # Thêm trường này
            'current_line': 0      # Thêm trường này
        })

        logger.info(f"Tạo task dịch văn bản mới: {task_id}")

        # Khởi chạy dịch trong thread riêng
        translator = GeminiTranslator(task_id=task_id, style_config=style_config)
        thread = Thread(
            target=process_text_translation,
            args=(task_id, translator, text)
        )
        thread.daemon = True
        thread.start()

        return jsonify({'task_id': task_id})

    except Exception as e:
        logger.error(f"Lỗi khi dịch văn bản: {str(e)}")
        return jsonify({'error': str(e)}), 500

def process_text_translation(task_id, translator, text):
    """Hàm xử lý dịch văn bản với bất kỳ định dạng nào"""
    try:
        task_manager.update_task(task_id, {
            'status': 'processing',
            'progress': 0,
            'translated_text': '',
            'total_lines': 1,
            'current_line': 0
        })

        logger.info(f"Bắt đầu dịch văn bản cho task {task_id}")

        # Phân tích văn bản thành các phần
        parts = []
        current_text = []
        
        for line in text.split('\n'):
            # Kiểm tra xem dòng có phải là định dạng đặc biệt không
            if ('-->' in line  # Định dạng thời gian SRT
                or line.strip().isdigit()  # Số thứ tự
                or not line.strip()):  # Dòng trống
                
                # Nếu có văn bản đang đợi dịch, xử lý nó trước
                if current_text:
                    parts.append({
                        'type': 'text',
                        'content': '\n'.join(current_text)
                    })
                    current_text = []
                
                # Thêm dòng định dạng vào parts
                if line.strip():
                    parts.append({
                        'type': 'format',
                        'content': line
                    })
                else:
                    parts.append({
                        'type': 'empty',
                        'content': ''
                    })
            else:
                # Dòng văn bản thường
                current_text.append(line)

        # Xử lý phần văn bản cuối cùng nếu có
        if current_text:
            parts.append({
                'type': 'text',
                'content': '\n'.join(current_text)
            })

        # Dịch các phần văn bản
        result_lines = []
        texts_to_translate = [p['content'] for p in parts if p['type'] == 'text']
        
        if texts_to_translate:
            translations = translator.translate_batch(texts_to_translate)
            trans_index = 0

            # Kết hợp lại kết quả, giữ nguyên định dạng
            for part in parts:
                if part['type'] == 'text':
                    if trans_index < len(translations):
                        result_lines.append(translations[trans_index])
                        trans_index += 1
                elif part['type'] == 'format':
                    result_lines.append(part['content'])
                elif part['type'] == 'empty':
                    result_lines.append('')

        # Cập nhật kết quả
        translated_text = '\n'.join(result_lines)
        task_manager.update_task(task_id, {
            'status': 'completed',
            'progress': 100,
            'translated_text': translated_text,
            'current_line': len(texts_to_translate)
        })
        
        logger.info(f"Hoàn thành dịch văn bản cho task {task_id}")

    except Exception as e:
        logger.error(f"Lỗi khi xử lý dịch văn bản: {str(e)}")
        task_manager.update_task(task_id, {
            'status': 'failed',
            'error': str(e),
            'translated_text': ''
        })

def split_subtitle_text(text, max_length=45, smart_split=False):
    """Tách văn bản phụ đề thành các dòng ngắn hơn, giữ nguyên câu"""
    if len(text) <= max_length:
        return [text]

    if smart_split:
        # Tách theo dấu câu và từ
        splits = []
        
        # Các dấu câu thường gặp
        sentence_endings = '.!?。！？'
        phrase_endings = ',，;；'
        
        # Tách câu thành các đoạn nhỏ
        def split_by_punctuation(text, max_length):
            parts = []
            current_part = ""
            
            words = text.split()
            for word in words:
                # Kiểm tra xem có thể thêm từ mới không
                test_part = (current_part + " " + word).strip()
                
                if len(test_part) <= max_length:
                    current_part = test_part
                    # Nếu từ kết thúc bằng dấu câu, tách thành đoạn mới
                    if any(word.endswith(p) for p in sentence_endings + phrase_endings):
                        parts.append(current_part)
                        current_part = ""
                else:
                    # Nếu phần hiện tại không rỗng, thêm vào kết quả
                    if current_part:
                        parts.append(current_part)
                    # Bắt đầu phần mới với từ hiện tại
                    if len(word) > max_length:
                        # Nếu từ quá dài, cắt thành các phần nhỏ hơn
                        while word:
                            parts.append(word[:max_length])
                            word = word[max_length:]
                    else:
                        current_part = word

            # Thêm phần còn lại nếu có
            if current_part:
                parts.append(current_part)
                
            return parts

        # Tách theo dấu câu trước
        parts = split_by_punctuation(text, max_length)
        
        # Xử lý từng phần
        for part in parts:
            if len(part) <= max_length:
                splits.append(part)
            else:
                # Nếu phần vẫn dài, tách theo từ
                subparts = split_by_punctuation(part, max_length)
                splits.extend(subparts)

        return splits if splits else [text]
    else:
        # Tách theo từ nếu không dùng smart split
        words = text.split()
        splits = []
        current_line = []
        current_length = 0
        
        for word in words:
            word_length = len(word)
            if current_length + len(current_line) + word_length <= max_length:
                current_line.append(word)
                current_length += word_length
            else:
                if current_line:
                    splits.append(' '.join(current_line))
                current_line = [word]
                current_length = word_length
        
        if current_line:
            splits.append(' '.join(current_line))
            
        return splits if splits else [text]
    
@app.route('/edit-subtitle', methods=['GET'])
def edit_subtitle_page():
    """Trang sửa phụ đề"""
    return render_template('edit.html')

@app.route('/edit-subtitle', methods=['POST']) 
def edit_subtitle():
    """API xử lý sửa phụ đề"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Không tìm thấy file'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Chưa chọn file'}), 400

        # Thêm timestamp vào tên file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = secure_filename(file.filename)
        base_name = Path(filename).stem
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{base_name}_{timestamp}.srt")
        file.save(filepath)

        # Tạo tên file output với timestamp
        output_filename = f"{base_name}_edited_{timestamp}.srt"
        output_filepath = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)

        # Lấy các tùy chọn từ form
        edit_config = {
            'remove_parentheses': request.form.get('remove_parentheses', 'true') == 'true',
            'check_chinese': request.form.get('check_chinese', 'true') == 'true',
            'clean_punctuation': request.form.get('clean_punctuation', 'true') == 'true',
            'use_ai': request.form.get('use_ai', 'true') == 'true',
            'style': request.form.get('style', 'standard')  # standard, formal, casual
        }

        # Tạo task mới
        task_id = task_manager.create_task({
            'filepath': filepath,
            'output_filepath': output_filepath,
            'edit_config': edit_config
        })

        # Xử lý trong thread riêng
        thread = Thread(
            target=process_edit_subtitle,
            args=(task_id, filepath, output_filepath, edit_config)
        )
        thread.daemon = True
        thread.start()

        return jsonify({'task_id': task_id})

    except Exception as e:
        logger.error(f"Lỗi khi sửa phụ đề: {str(e)}")
        return jsonify({'error': str(e)}), 500

def process_edit_subtitle(task_id, input_file, output_file, config):
    """Hàm xử lý sửa phụ đề theo batch"""
    try:
        # Đọc file input
        subs = pysrt.open(input_file, encoding='utf-8')
        total_subs = len(subs)
        
        # Khởi tạo danh sách phụ đề mới
        new_subs = pysrt.SubRipFile()
        
        task_manager.update_task(task_id, {
            'total_subs': total_subs,
            'current_sub': 0,
            'progress': 0
        })

        # Chia thành các batch 20 phụ đề
        batch_size = 20
        batches = [subs[i:i+batch_size] for i in range(0, len(subs), batch_size)]

        # Xử lý từng batch
        for batch_idx, batch in enumerate(batches):
            # Thu thập text từ batch hiện tại
            batch_texts = []
            for sub in batch:
                text = sub.text.strip()
                if text:
                    batch_texts.append(text)

            # Sửa tất cả text trong batch
            edited_texts = []
            for text in batch_texts:
                edited_text = edit_subtitle_text(text, config)
                edited_texts.append(edited_text)

            # Cập nhật phụ đề với text đã sửa
            text_idx = 0
            for sub in batch:
                if sub.text.strip():
                    new_sub = pysrt.SubRipItem(
                        index=len(new_subs) + 1,
                        start=sub.start,
                        end=sub.end,
                        text=edited_texts[text_idx]
                    )
                    text_idx += 1
                else:
                    new_sub = pysrt.SubRipItem(
                        index=len(new_subs) + 1,
                        start=sub.start,
                        end=sub.end,
                        text=''
                    )
                new_subs.append(new_sub)

            # Cập nhật tiến độ
            progress = min(100, int((batch_idx + 1) * batch_size * 100 / total_subs))
            task_manager.update_task(task_id, {
                'progress': progress,
                'current_sub': min((batch_idx + 1) * batch_size, total_subs)
            })

            # Log tiến độ
            logger.info(f"Đã xử lý {min((batch_idx + 1) * batch_size, total_subs)}/{total_subs} phụ đề")

            # Delay nhỏ giữa các batch
            time.sleep(0.1)

        # Lưu file kết quả
        new_subs.save(output_file, encoding='utf-8')
        
        task_manager.update_task(task_id, {
            'status': 'completed',
            'progress': 100
        })

        logger.info(f"Hoàn thành sửa phụ đề: {output_file}")

    except Exception as e:
        logger.error(f"Lỗi khi xử lý sửa phụ đề: {str(e)}")
        task_manager.update_task(task_id, {
            'status': 'failed',
            'error': str(e)
        })

def fix_subtitle_style_with_ai(text, style):
    """Sử dụng AI để điều chỉnh phong cách phụ đề"""
    try:
        # Tạo instance GeminiTranslator với style config phù hợp
        style_config = {
            "tone": style,
            "audience": "general",
            "creativity": 0.3
        }
        translator = GeminiTranslator(style_config=style_config)

        prompt = f"""Chỉnh sửa phụ đề theo phong cách sau:
Phong cách: {style}
- formal: văn phong trang trọng, lịch sự, dùng từ ngữ chuẩn mực
- casual: thân mật, tự nhiên, có thể dùng từ lóng phổ biến
- standard: trung tính, dễ hiểu, phù hợp đại chúng

Phụ đề gốc: {text}

Yêu cầu:
1. Giữ nguyên ý nghĩa gốc
2. Điều chỉnh từ ngữ phù hợp phong cách
3. Xóa các dấu thừa (..., !!!)
4. Không thêm giải thích hay ghi chú
5. Trả về phụ đề đã chỉnh sửa

Output format: Chỉ trả về phụ đề đã sửa"""

        response = translator._call_api({
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "temperature": 0.3,
                "topK": 1,
                "topP": 1,
                "maxOutputTokens": 1024
            }
        })

        if not response or 'candidates' not in response:
            logger.warning(f"Không nhận được phản hồi từ API cho phụ đề: {text[:50]}...")
            return text

        # Lấy kết quả và làm sạch
        result = response['candidates'][0]['content']['parts'][0]['text']
        result = result.strip()
        
        if not result:
            logger.warning("Nhận được kết quả rỗng từ API")
            return text
            
        # Log để debug
        logger.debug(f"Input: {text}")
        logger.debug(f"Style: {style}")
        logger.debug(f"Output: {result}")

        return result

    except Exception as e:
        logger.error(f"Lỗi khi sửa phong cách: {str(e)}")
        return text  # Trả về text gốc nếu có lỗi

def edit_subtitle_text(text, config):
    """Sửa text phụ đề theo config"""
    try:
        # Xóa text trong ngoặc ()
        if config['remove_parentheses']:
            text = re.sub(r'\([^)]*\)', '', text)
            text = re.sub(r'（[^）]*）', '', text)  # Ngoặc tiếng Trung

        # Xử lý emoji và các ký tự đặc biệt
        if config['clean_punctuation']:
            # Xóa emoji và icon
            text = remove_emojis(text)
            # Xóa các ký tự đặc biệt không mong muốn
            text = re.sub(r'[☆★♪♫♬〜~]+', '', text)
            # Xóa các dấu câu lặp
            text = re.sub(r'[.]{3,}', '...', text)  # Giữ lại tối đa 3 dấu chấm
            text = re.sub(r'!{2,}', '!', text)      # Giữ lại 1 dấu chấm than
            text = re.sub(r'\?{2,}', '?', text)     # Giữ lại 1 dấu hỏi
            # Chuẩn hóa khoảng trắng
            text = re.sub(r'\s+', ' ', text)
            # Xóa khoảng trắng đầu/cuối dòng
            text = text.strip()

        # Kiểm tra và xử lý chữ Trung
        if config['check_chinese']:
            chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
            if chinese_chars:
                logger.warning(f"Phát hiện chữ Trung: {''.join(chinese_chars)}")
                # Thay thế chữ Trung bằng khoảng trắng
                text = re.sub(r'[\u4e00-\u9fff]+', ' ', text)
                # Chuẩn hóa lại khoảng trắng
                text = re.sub(r'\s+', ' ', text).strip()

        # Xử lý style với AI nếu được bật
        if config['use_ai'] and text.strip():
            text = fix_subtitle_style_with_ai(text, config['style'])

        # Xử lý cuối cùng
        text = text.strip()
        # Đảm bảo có dấu câu kết thúc
        if text and not text[-1] in '.!?':
            text += '.'
        # Viết hoa chữ đầu câu
        if text:
            text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()

        return text

    except Exception as e:
        logger.error(f"Lỗi khi sửa text: {str(e)}")
        return text

def remove_emojis(text):
    """Xóa emoji và các ký tự đặc biệt"""
    # Mở rộng pattern để bắt thêm nhiều loại emoji và icon
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        u"\U0001F004-\U0001F0CF"  # Additional emoticons
        u"\U0001F170-\U0001F251"  # Enclosed characters
        u"\U00002600-\U000026FF"  # Miscellaneous Symbols
        u"\U00002700-\U000027BF"  # Dingbats
        "]+", flags=re.UNICODE)
    
    # Xóa emoji
    text = emoji_pattern.sub('', text)
    
    # Xóa các ký tự đặc biệt khác
    text = re.sub(r'[¯\\_(ツ)_/¯♡♥❤️]+', '', text)
    
    return text

@app.route('/summarize', methods=['GET'])
def summarize_page():
    """Trang tóm tắt phụ đề"""
    return render_template('summarize.html')

@app.route('/summarize-subtitle', methods=['POST'])
def summarize_subtitle():
    """API xử lý tóm tắt phụ đề"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Không tìm thấy file'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Chưa chọn file'}), 400

        # Lưu file với timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{filename}_{timestamp}.srt")
        file.save(filepath)

        # Lấy các tùy chọn
        summary_config = {
            'length': request.form.get('length', 'medium'),
            'style': request.form.get('style', 'paragraph')
        }

        # Tạo task mới
        task_id = task_manager.create_task({
            'filepath': filepath,
            'summary_config': summary_config,
            'status': 'processing'
        })

        # Xử lý trong thread riêng
        thread = Thread(
            target=process_summarize_subtitle,
            args=(task_id, filepath, summary_config)
        )
        thread.daemon = True
        thread.start()

        return jsonify({'task_id': task_id})

    except Exception as e:
        logger.error(f"Lỗi khi tóm tắt phụ đề: {str(e)}")
        return jsonify({'error': str(e)}), 500

def process_summarize_subtitle(task_id, input_file, config):
    """Hàm xử lý tóm tắt phụ đề"""
    try:
        # Đọc file input
        subs = pysrt.open(input_file, encoding='utf-8')
        total_subs = len(subs)
        
        task_manager.update_task(task_id, {
            'total_subs': total_subs,
            'current_sub': 0,
            'progress': 0
        })

        # Gom nội dung phụ đề
        all_text = []
        for sub in subs:
            text = sub.text.strip()
            if text:
                all_text.append(text)

        # Khởi tạo AI để tóm tắt
        summarizer = GeminiTranslator(task_id=task_id)
        
        # Tạo prompt dựa trên config
        length_guide = {
            'short': '2-3 câu ngắn gọn',
            'medium': '4-6 câu chi tiết vừa phải',
            'long': '7-10 câu chi tiết'
        }
        
        style_guide = {
            'bullet': 'dạng điểm chính, mỗi ý một dòng',
            'paragraph': 'dạng đoạn văn liền mạch',
            'timeline': 'theo trình tự thời gian diễn biến'
        }

        prompt = f"""Tóm tắt nội dung phụ đề sau:

Nội dung gốc:
{' '.join(all_text)}

Yêu cầu:
1. Độ dài: {length_guide[config['length']]}
2. Phong cách: {style_guide[config['style']]}
3. Giữ các thông tin quan trọng
4. Viết rõ ràng, dễ hiểu
5. Chỉ trả về nội dung tóm tắt, không có giải thích

Format: Trả về nội dung tóm tắt theo yêu cầu"""

        # Gọi API tóm tắt
        response = summarizer._call_api({
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "temperature": 0.3,
                "topK": 1,
                "topP": 1,
                "maxOutputTokens": 2048
            }
        })

        if not response or 'candidates' not in response:
            raise Exception("Không nhận được phản hồi từ AI")

        # Lấy kết quả tóm tắt
        summary = response['candidates'][0]['content']['parts'][0]['text'].strip()

        # Format kết quả theo style
        if config['style'] == 'bullet':
            # Thêm dấu bullet nếu chưa có
            if not any(line.startswith(('•', '-', '*')) for line in summary.split('\n')):
                summary = '\n'.join(f'• {line}' for line in summary.split('\n'))
        
        # Cập nhật kết quả
        task_manager.update_task(task_id, {
            'status': 'completed',
            'progress': 100,
            'summary': summary
        })

        logger.info(f"Hoàn thành tóm tắt phụ đề: {input_file}")

    except Exception as e:
        logger.error(f"Lỗi khi xử lý tóm tắt phụ đề: {str(e)}")
        task_manager.update_task(task_id, {
            'status': 'failed',
            'error': str(e)
        })
# Khởi động cleanup thread
cleanup_thread = Thread(target=cleanup_old_files)
cleanup_thread.daemon = True
cleanup_thread.start()

if __name__ == '__main__':
    app.run(debug=True)