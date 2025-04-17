import random
from flask import request
import requests
import json
import pysrt
import logging
import threading
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import re
import os
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.shared import add_user_request, get_user_rate_limit

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load biến môi trường
load_dotenv()
tasks = {}


class RateLimiter:
    def __init__(self, requests_per_min=15):
        self.requests_per_min = requests_per_min
        self.lock = threading.Lock()
        self.task_id = None
        self.user_id = None

    def wait_if_needed(self):
        with self.lock:
            now = datetime.now()
            requests_count = get_user_rate_limit(self.user_id)
            
            if requests_count >= self.requests_per_min:
                sleep_time = 60
                message = f"Đang chờ {sleep_time:.1f} giây do giới hạn tốc độ API..."
                logger.warning(f"Rate limit exceeded for user {self.user_id}, retrying after {sleep_time}s")
                
                if self.task_id and self.task_id in tasks:
                    tasks[self.task_id].update({
                        'rate_limited': True,
                        'rate_limit_message': message,
                        'rate_limit_time': sleep_time,
                        'user_id': self.user_id
                    })
                time.sleep(sleep_time)
            
            add_user_request(self.user_id)

    def set_task_id(self, task_id):
        self.task_id = task_id

class APIKeyManager:
    def __init__(self):
        self.keys = {}
        self.lock = threading.Lock()
        # Reset tất cả các key khi khởi tạo
        self.load_keys_from_env()
        
    def load_keys_from_env(self):
        """Load và reset tất cả API keys"""
        for i in range(1, 3):  # Đọc 2 key
            key = os.getenv(f'GEMINI_API_KEY_{i}')
            if key:
                self.keys[key] = {
                    'requests': 0,  # Reset số request
                    'last_reset': datetime.now(),
                    'is_active': True,  # Đặt lại trạng thái active
                    'errors': 0  # Reset số lỗi
                }
        logger.info(f"Loaded and reset {len(self.keys)} API keys")

    def get_available_key(self):
        """Lấy key khả dụng với kiểm tra và reset tự động"""
        with self.lock:
            now = datetime.now()
            
            # Reset các key đã quá 1 phút
            for key_info in self.keys.values():
                if (now - key_info['last_reset']).total_seconds() >= 60:
                    key_info['requests'] = 0
                    key_info['last_reset'] = now
                    key_info['is_active'] = True
                    key_info['errors'] = 0

            # Lọc các key còn khả dụng
            available_keys = [
                key for key, info in self.keys.items() 
                if info['is_active'] and info['requests'] < 15
            ]
            
            if not available_keys:
                logger.error(f"No available keys. Total keys: {len(self.keys)}")
                logger.info("Key status:")
                for key, info in self.keys.items():
                    logger.info(f"Key {key[:10]}...: requests={info['requests']}, active={info['is_active']}, errors={info['errors']}")
                return None
            
            selected_key = random.choice(available_keys)
            self.keys[selected_key]['requests'] += 1
            return selected_key

    def mark_key_error(self, api_key):
        """Đánh dấu key lỗi với reset tự động"""
        with self.lock:
            if api_key in self.keys:
                now = datetime.now()
                key_info = self.keys[api_key]
                
                # Reset key nếu đã quá 1 phút
                if (now - key_info['last_reset']).total_seconds() >= 60:
                    key_info['requests'] = 0
                    key_info['last_reset'] = now
                    key_info['is_active'] = True
                    key_info['errors'] = 0
                else:
                    key_info['errors'] += 1
                    if key_info['errors'] >= 3:
                        key_info['is_active'] = False

class GeminiTranslator:
    def __init__(self, task_id=None, style_config=None):
        self.key_manager = APIKeyManager()
        self.session = requests.Session()
        self.rate_limiter = RateLimiter(requests_per_min=15)
        if task_id:
            self.rate_limiter.task_id = task_id
        self.max_input_length = 100000  # Tăng max input length
        self.batch_size = 20  # Cố định batch size là 20
        self.max_workers = 2  # Giảm số workers để tránh quá tải API
        self.chunk_delay = 1  # Tăng delay giữa các chunk
        self.use_smart_batching = True  # Bật tính năng smart batching
        self.cache_file = "translation_cache.json"
        self.cache = self.load_cache()
        self.task_id = task_id  # Thêm task_id nếu cần
        self.style_config = style_config or {
            "tone": "standard",  # standard, casual, formal, humorous
            "audience": "general",  # general, youth, professional
            "creativity": 0.1,  # 0.1 - 1.0
            "source_lang": "zh",  # Mặc định là tiếng Trung
            "target_lang": "vi"   # Mặc định là tiếng Việt
        }
        self.use_cache = style_config.get('use_cache', False) if style_config else False
        self.split_config = style_config.get('split_config', {
            'enabled': False,
            'max_length': 45,
            "preserve_sentences": request.form.get('preserve_sentences', 'true') == 'true'
        }) if style_config else None
        self.max_retries = 5
        self.min_backoff = 2
        self.max_backoff = 60

    def load_cache(self):
        """Load translation cache from file with error handling"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:  # Only parse if file is not empty
                        return json.loads(content)
            # If file doesn't exist or is empty, create new cache
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump({}, f, ensure_ascii=False)
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Error loading cache file: {str(e)}")
            # Backup corrupted cache file
            if os.path.exists(self.cache_file):
                backup_file = f"{self.cache_file}.bak"
                os.rename(self.cache_file, backup_file)
                logger.info(f"Corrupted cache file backed up to {backup_file}")
            # Create new empty cache
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump({}, f, ensure_ascii=False)
            return {}

    def save_cache(self):
        """Save translation cache to file with error handling"""
        try:
            # Create temp file first
            temp_file = f"{self.cache_file}.tmp"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
            
            # If successful, replace original file
            if os.path.exists(self.cache_file):
                os.remove(self.cache_file)
            os.rename(temp_file, self.cache_file)
        except Exception as e:
            logger.error(f"Error saving cache file: {str(e)}")
            # Try to remove temp file if it exists
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass

    def get_style_prompt(self):
        style_prompts = {
            "tone": {
                "standard": "Sử dụng ngôn ngữ trung tính, tự nhiên, không thêm icon hoặc chú thích",
                "casual": "Sử dụng ngôn ngữ thân mật, gần gũi, bắt trend, có thể dùng từ lóng phổ biến, không thêm icon hoặc chú thích",
                "formal": "Sử dụng ngôn ngữ trang trọng, lịch sự, không thêm icon hoặc chú thích",
                "humorous": "Thêm yếu tố hài hước khi phù hợp, giọng điệu vui tươi, không thêm icon hoặc chú thích"
            },
            "audience": {
                "general": "Phù hợp với mọi đối tượng",
                "youth": "Sử dụng ngôn ngữ trẻ trung, hiện đại, dễ tiếp cận với giới trẻ",
                "professional": "Sử dụng thuật ngữ chuyên ngành khi cần thiết"
            }
        }

        style_text = f"""Phong cách dịch:
- {style_prompts['tone'][self.style_config['tone']]}
- {style_prompts['audience'][self.style_config['audience']]}
- Mức độ sáng tạo: {self.style_config['creativity']}"""
        return style_text

    def _call_api(self, payload):
        """Call API with detailed logging"""
        try:
            api_key = self.key_manager.get_available_key()
            if not api_key:
                logger.error("Key status:")
                for key, info in self.key_manager.keys.items():
                    logger.error(f"""
                    Key: {key[:10]}...
                    Requests: {info['requests']}
                    Active: {info['is_active']}
                    Errors: {info['errors']}
                    Last Reset: {info['last_reset']}
                    """)
                raise ValueError("No available API keys")
                
            logger.info(f"Gọi API với payload size: {len(str(payload))} bytes")
            
            self.session.headers.update({
                'x-goog-api-key': api_key,
                'Content-Type': 'application/json'
            })
            
            response = self.session.post(
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
                json=payload,
                timeout=30
            )

            logger.info(f"API Status Code: {response.status_code}")
            
            if response.status_code == 429:
                logger.warning(f"Rate limit exceeded for key {api_key}")
                self.key_manager.mark_key_error(api_key)
                time.sleep(2)
                return self._call_api(payload)

            response.raise_for_status()
            return response.json()

        except Exception as e:
            logger.error(f"API call failed: {str(e)}")
            self.key_manager.mark_key_error(api_key)
            raise

    def translate_batch(self, texts):
        """Dịch một batch các phụ đề"""
        try:
            if not texts:
                return []

            texts = [text.strip() for text in texts if text.strip()]

            # Tạo prompt style từng phần riêng biệt
            style_details = self.get_style_prompt().split('\n')
            tone_desc = style_details[1].strip()
            audience_desc = style_details[2].strip()

            # Tạo prompt không sử dụng backslash
            prompt = (
                f"Translate {self.style_config['source_lang']} to {self.style_config['target_lang']} with the following style:\n"
                f"- Tone: {self.style_config['tone']} ({tone_desc})\n"
                f"- Audience: {self.style_config['audience']} ({audience_desc})\n"
                f"- Creativity: {self.style_config['creativity']}\n\n"
                f"Input:\n"
                f"{chr(10).join(f'[{i+1}] {text}' for i, text in enumerate(texts))}\n\n"
                f"Output format:\n"
                f"- Just translate the text without adding any icons, emojis, or additional comments.\n"
                f"- Do not add any humor, jokes, or annotations unless explicitly present in the original text.\n"
                f"- Only return the translated text for each line."
            )

            response = self._call_api({
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }],
                "generationConfig": {
                    "temperature": float(self.style_config['creativity']),
                    "topK": 1,
                    "topP": 1,
                    "maxOutputTokens": 2048
                }
            })

            if not response:
                return [""] * len(texts)

            # Chỉ lấy các dòng bắt đầu bằng số hoặc [số]
            translations = []
            result_text = response['candidates'][0]['content']['parts'][0]['text']
            
            for line in result_text.split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                # Chỉ xử lý các dòng bắt đầu bằng số hoặc [số]
                match = re.match(r'(?:\[)?(\d+)(?:\])?\s*[:：]?\s*(.*)', line)
                if match and match.group(2):
                    translations.append(match.group(2).strip())
                elif not any(line.startswith(x) for x in ["Here's", "Output", "Input"]):
                    # Nếu không có số nhưng cũng không phải dòng giới thiệu
                    translations.append(line)

            # Đảm bảo số lượng bản dịch khớp với đầu vào
            if len(translations) != len(texts):
                logger.warning(f"Số lượng bản dịch ({len(translations)}) không khớp với số lượng cần dịch ({len(texts)})")
                while len(translations) < len(texts):
                    translations.append("")

            return translations[:len(texts)]

        except Exception as e:
            logger.error(f"Lỗi khi dịch batch: {str(e)}")
            return [""] * len(texts)

    def split_long_subtitle(self, text, max_length=45):
        """Tách phụ đề dài thành các dòng ngắn hơn, giữ nguyên câu"""
        if len(text) <= max_length:
            return [text]

        # Tách văn bản thành các câu dựa trên dấu câu
        sentences = re.split(r'([.!?])', text)  # Tách theo dấu câu (., !, ?)
        sentences = [''.join(pair).strip() for pair in zip(sentences[0::2], sentences[1::2])]  # Ghép lại câu và dấu câu

        lines = []
        current_line = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)
            space_length = 1 if current_line else 0

            if current_length + sentence_length + space_length <= max_length:
                current_line.append(sentence)
                current_length += sentence_length + space_length
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [sentence]
                current_length = sentence_length

        if current_line:
            lines.append(' '.join(current_line))

        return lines

    def _process_batch(self, batch, indices, subs):
        try:
            translations = self.translate_batch(batch)
            if len(translations) != len(batch):
                logger.warning(f"Translation mismatch: got {len(translations)}, expected {len(batch)}")
                translations.extend([batch[i] for i in range(len(translations), len(batch))])

            current_idx = 0
            for idx, trans, orig in zip(indices, translations, batch):
                if idx < len(subs):
                    # Kiểm tra xem có cần tách phụ đề không
                    if self.split_config and self.split_config['enabled']:
                        max_length = self.split_config['max_length']
                        split_lines = self.split_long_subtitle(trans, max_length)

                        if len(split_lines) > 1:
                            # Tính thời gian cho mỗi dòng
                            duration = subs[idx].end.ordinal - subs[idx].start.ordinal
                            time_per_line = duration / len(split_lines)

                            # Cập nhật dòng đầu tiên
                            subs[idx].text = split_lines[0]
                            self.cache[orig] = split_lines[0]

                            # Thêm các dòng phụ đề mới
                            for i, line in enumerate(split_lines[1:], 1):
                                new_start = subs[idx].start.ordinal + (i * time_per_line)
                                new_end = new_start + time_per_line

                                new_sub = pysrt.SubRipItem(
                                    index=len(subs) + current_idx,
                                    start=pysrt.SubRipTime.from_ordinal(int(new_start)),
                                    end=pysrt.SubRipTime.from_ordinal(int(new_end)),
                                    text=line
                                )
                                subs.append(new_sub)
                                current_idx += 1
                        else:
                            subs[idx].text = trans
                            self.cache[orig] = trans
                    else:
                        subs[idx].text = trans
                        self.cache[orig] = trans

        except Exception as e:
            logger.error(f"Batch processing failed: {str(e)}")
            for idx, orig in zip(indices, batch):
                if idx < len(subs):
                    subs[idx].text = orig

    def translate_srt_file(self, input_file, output_file, source_lang="zh", target_lang="vi", progress_callback=None):
        logger.info(f"Bắt đầu dịch file {input_file}")
        subs = pysrt.open(input_file, encoding='utf-8')
        total_subs = len(subs)
        processed = 0

        # Tối ưu việc chia batch
        smart_batches = self._create_smart_batches(subs)
        total_batches = len(smart_batches)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            for batch_idx, (batch_texts, batch_indices) in enumerate(smart_batches):
                # Kiểm tra cache trước khi submit
                uncached_texts = []
                uncached_indices = []
                
                for text, idx in zip(batch_texts, batch_indices):
                    if self.use_cache and text in self.cache:
                        subs[idx].text = self.cache[text]
                        processed += 1
                    else:
                        uncached_texts.append(text)
                        uncached_indices.append(idx)

                if uncached_texts:
                    # Submit batch chưa được cache
                    future = executor.submit(
                        self._process_batch_with_rate_limit,
                        uncached_texts,
                        uncached_indices,
                        subs,
                        batch_idx,
                        total_batches
                    )
                    futures.append(future)

                # Cập nhật tiến độ
                if progress_callback:
                    progress = min(100, int(processed * 100 / total_subs))
                    progress_callback(progress)

                # Smart delay giữa các batch để tránh rate limit
                if batch_idx < total_batches - 1:
                    time.sleep(self.chunk_delay)

            # Thu thập kết quả
            for future in as_completed(futures):
                try:
                    processed += future.result()
                except Exception as e:
                    logger.error(f"Batch processing failed: {str(e)}")

        # Lưu kết quả
        subs.save(output_file, encoding='utf-8')
        if self.use_cache:
            self.save_cache()
            
        logger.info(f"Hoàn thành dịch file: {processed}/{total_subs}")
        return processed == total_subs

    def _create_smart_batches(self, subs):
        """Tạo batches với kích thước cố định 20"""
        batches = []
        current_texts = []
        current_indices = []
        
        for i, sub in enumerate(subs):
            text = sub.text.strip()
            if not text:
                continue

            current_texts.append(text)
            current_indices.append(i)
            
            # Khi đủ 20 dòng hoặc là dòng cuối, tạo batch mới
            if len(current_texts) == 20 or i == len(subs) - 1:
                batches.append((current_texts[:], current_indices[:]))
                current_texts = []
                current_indices = []

        return batches

    def _process_batch_with_rate_limit(self, texts, indices, subs, batch_idx, total_batches):
        """Xử lý batch với kiểm soát rate limit"""
        try:
            # Đợi rate limit nếu cần
            self.rate_limiter.wait_if_needed()

            # Dịch batch
            translations = self.translate_batch(texts)
            processed = 0

            # Cập nhật subtitles
            for idx, trans, orig in zip(indices, translations, texts):
                if idx < len(subs):
                    subs[idx].text = trans
                    if self.use_cache:
                        self.cache[orig] = trans
                    processed += 1

            return processed

        except Exception as e:
            logger.error(f"Error processing batch {batch_idx + 1}/{total_batches}: {str(e)}")
            # Fallback: giữ nguyên text gốc
            for idx, text in zip(indices, texts):
                if idx < len(subs):
                    subs[idx].text = text
            return 0

    def _parse_translations(self, response, expected_count):
        """Parse kết quả từ Gemini API"""
        try:
            if not response or 'candidates' not in response or not response['candidates']:
                logger.error("Không nhận được phản hồi hợp lệ từ API")
                return []

            # Lấy text từ phản hồi API
            text = response['candidates'][0]['content']['parts'][0]['text']
            
            # Tách các dòng dịch
            translations = []
            lines = text.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Tìm và lấy phần dịch (sau dấu :]
                match = re.match(r'\[?\d+\]?\s*[:：]?\s*(.*)', line)
                if match:
                    translation = match.group(1).strip()
                    if translation:
                        translations.append(translation)
                elif line:  # Nếu không match pattern nhưng có nội dung
                    translations.append(line)

            # Kiểm tra số lượng bản dịch
            if len(translations) != expected_count:
                logger.warning(f"Số lượng bản dịch ({len(translations)}) không khớp với số lượng cần dịch ({expected_count})")
                # Điền thêm các bản dịch trống nếu thiếu
                while len(translations) < expected_count:
                    translations.append("")

            return translations[:expected_count]  # Chỉ trả về đúng số lượng cần

        except Exception as e:
            logger.error(f"Lỗi khi parse kết quả dịch: {str(e)}")
            return [""] * expected_count  # Trả về list rỗng với độ dài mong muốn

def main():
    try:
        translator = GeminiTranslator()  # Không truyền task_id
        input_file = "phudetrung.srt"
        output_file = "phudetrung_translated.srt"
        
        success = translator.translate_srt_file(input_file, output_file)
        if not success:
            logger.warning("File chưa được dịch hoàn toàn, kiểm tra log để biết chi tiết.")
    except Exception as e:
        logger.error(f"Lỗi: {str(e)}")

if __name__ == "__main__":
    main()