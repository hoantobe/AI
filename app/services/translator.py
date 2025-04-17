from app.utils.rate_limiter import RateLimiter
from app.utils.logger import setup_logger
import requests
import time
import os
from dotenv import load_dotenv
import pysrt

# Assuming tasks is a global dictionary to track task statuses
tasks = {}

logger = setup_logger(__name__)
load_dotenv()

class TranslatorService:
    def __init__(self, task_id=None):
        # Lấy API key từ biến môi trường
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY không được cấu hình trong file .env")
            
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
        self.headers = {'Content-Type': 'application/json'}
        self.rate_limiter = RateLimiter(requests_per_min=12)
        self.task_id = task_id

    def translate_text(self, text, source_lang="zh", target_lang="vi"):
        """Translate text using Gemini API with proper error handling and retries"""
        try:
            self.rate_limiter.wait_if_needed()
            
            if not text or not self._is_source_language(text, source_lang):
                return text
                
            response = self._call_api(text)
            
            if response.status_code == 200:
                return self._parse_response(response.json())
            elif response.status_code == 429:
                retry_after = self._get_retry_after(response)
                logger.warning(f"Rate limit exceeded. Waiting {retry_after}s...")
                time.sleep(retry_after)
                return self.translate_text(text, source_lang, target_lang)
            else:
                logger.error(f"API error {response.status_code}: {response.text}")
                return text
                
        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            return text
            
    def _is_source_language(self, text, lang):
        """Validate text is in source language"""
        if lang == "zh":
            return any('\u4e00' <= char <= '\u9fff' for char in text)
        return True
        
    def _call_api(self, text):
        """Make API call with proper formatting"""
        payload = {
            "contents": [{
                "parts": [{
                    "text": f"""Act as a professional Chinese-Vietnamese translator.
Translate this Chinese subtitle to Vietnamese. Return ONLY the translation, no explanations:

Chinese: {text}

Translation:"""
                }]
            }]
        }
        return requests.post(
            self.api_url,
            headers=self.headers,
            json=payload
        )
        
    def _parse_response(self, response):
        """Parse API response and extract translation"""
        try:
            translated = response['candidates'][0]['content']['parts'][0]['text'].strip()
            if "Translation:" in translated:
                translated = translated.split("Translation:")[-1].strip()
            return translated
        except:
            logger.error("Failed to parse API response")
            return None

    def _get_retry_after(self, response):
        """Get retry after duration from response"""
        try:
            details = response.json().get('error', {}).get('details', [])
            for detail in details:
                if '@type' == 'type.googleapis.com/google.rpc.RetryInfo':
                    return int(detail.get('retryDelay', '60s').replace('s', ''))
        except:
            pass
        return 60  # Default retry after 60s

    def translate_srt_file(self, input_file, output_file, source_lang="zh", target_lang="vi", progress_callback=None):
        """Translate SRT file with batch processing"""
        try:
            subs = pysrt.open(input_file, encoding='utf-8')
            total = len(subs)
            batch_size = 5  # Số câu xử lý mỗi lần
            
            for i in range(0, total, batch_size):
                batch = subs[i:min(i+batch_size, total)]
                
                # Kiểm tra tạm dừng
                if self.task_id and tasks.get(self.task_id, {}).get('is_paused'):
                    while tasks.get(self.task_id, {}).get('is_paused'):
                        time.sleep(1)

                # Xử lý batch
                for sub in batch:
                    if sub.text.strip():
                        sub.text = self.translate_text(sub.text)

                # Lưu file sau mỗi batch
                subs.save(output_file, encoding='utf-8')
                
                # Cập nhật tiến độ
                if progress_callback:
                    progress = min(100, int((i + batch_size) / total * 100))
                    progress_callback(progress)

                # Rate limiting
                time.sleep(2)

            # Hoàn thành
            if self.task_id and self.task_id in tasks:
                tasks[self.task_id]['status'] = 'completed'
                tasks[self.task_id]['progress'] = 100

        except Exception as e:
            logger.error(f"File translation error: {str(e)}")
            if self.task_id and self.task_id in tasks:
                tasks[self.task_id]['status'] = 'failed'
                tasks[self.task_id]['error'] = str(e)
            raise