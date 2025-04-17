import pysrt
import requests
import os
import time
from pydub import AudioSegment
import logging
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
from concurrent.futures import ThreadPoolExecutor, as_completed

# Cấu hình logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load API key từ .env
load_dotenv()
FPT_API_KEY = os.getenv('FPT_API_KEY')
if not FPT_API_KEY:
    raise ValueError("FPT_API_KEY không được cấu hình trong file .env")

class SubtitleTTS:
    def __init__(self, srt_file, voice="leminh", speed=1.0, output_file="full_audio.mp3"):
        self.srt_file = srt_file
        self.voice = voice
        self.speed = speed
        self.subs = pysrt.open(srt_file, encoding='utf-8')
        self.output_dir = "tts_output"
        self.output_file = output_file
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=30))
    def download_audio(self, audio_url, output_file):
        logger.debug(f"Đang tải file từ: {audio_url}")
        audio_response = requests.get(audio_url, timeout=10)
        if audio_response.status_code == 429:  # Too Many Requests
            retry_after = int(audio_response.headers.get("Retry-After", 10))
            logger.warning(f"Rate limit exceeded, chờ {retry_after}s")
            time.sleep(retry_after)
            raise requests.exceptions.RequestException("Rate limit")
        audio_response.raise_for_status()
        with open(output_file, 'wb') as f:
            f.write(audio_response.content)
        logger.info(f"Đã tải file âm thanh: {output_file}")

    def text_to_speech(self, text, output_file):
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            logger.debug(f"File đã tồn tại và hợp lệ: {output_file}")
            return

        url = "https://api.fpt.ai/hmi/tts/v5"
        headers = {
            "api-key": FPT_API_KEY,
            "voice": self.voice,
            "speed": str(self.speed),
            "Content-Type": "application/json"
        }
        payload = text.encode('utf-8')

        try:
            response = requests.post(url, headers=headers, data=payload)
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 10))
                logger.warning(f"Rate limit exceeded, chờ {retry_after}s")
                time.sleep(retry_after)
                raise requests.exceptions.RequestException("Rate limit on POST")
            response.raise_for_status()
            result = response.json()
            logger.debug(f"Phản hồi từ FPT AI: {result}")

            audio_url = result.get("async")
            if not audio_url:
                raise ValueError("Không nhận được URL âm thanh từ FPT AI")

            max_wait = 20
            wait_interval = 2
            elapsed = 0
            while elapsed < max_wait:
                try:
                    self.download_audio(audio_url, output_file)
                    return
                except requests.exceptions.RequestException as e:
                    if e.response and e.response.status_code == 404:
                        logger.info(f"File chưa sẵn sàng, chờ {wait_interval}s...")
                        time.sleep(wait_interval)
                        elapsed += wait_interval
                    else:
                        raise

            raise ValueError(f"File âm thanh không sẵn sàng sau {max_wait}s: {audio_url}")

        except Exception as e:
            logger.error(f"Lỗi khi tạo TTS: {str(e)}")
            raise

    def generate_and_combine_audio(self):
        logger.info("Bắt đầu tạo và ghép âm thanh...")
        full_audio = AudioSegment.empty()
        last_end_seconds = 0
        temp_files = []

        # Tạo file âm thanh song song (giảm max_workers để tránh rate limit)
        with ThreadPoolExecutor(max_workers=2) as executor:  # Giảm xuống 2 worker
            futures = []
            for i, sub in enumerate(self.subs):
                text = sub.text.strip()
                if not text:
                    continue
                temp_file = os.path.join(self.output_dir, f"temp_{i}.mp3")
                futures.append(executor.submit(self.text_to_speech, text, temp_file))
                temp_files.append((i, temp_file, sub))
            
            # Kiểm tra lỗi từ các future
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Lỗi trong luồng: {str(e)}")
                    raise

        # Ghép file
        for i, temp_file, sub in sorted(temp_files, key=lambda x: x[0]):
            start_seconds = (sub.start.hours * 3600 + sub.start.minutes * 60 + 
                            sub.start.seconds + sub.start.milliseconds / 1000)
            end_seconds = (sub.end.hours * 3600 + sub.end.minutes * 60 + 
                          sub.end.seconds + sub.end.milliseconds / 1000)
            if not os.path.exists(temp_file) or os.path.getsize(temp_file) == 0:
                logger.warning(f"File {temp_file} không hợp lệ, bỏ qua")
                continue

            audio_segment = AudioSegment.from_mp3(temp_file)
            silence_duration = (start_seconds - last_end_seconds) * 1000
            if silence_duration > 0:
                full_audio += AudioSegment.silent(duration=silence_duration)
            full_audio += audio_segment
            last_end_seconds = end_seconds
            os.remove(temp_file)

        full_audio.export(self.output_file, format="mp3")
        logger.info(f"Đã tạo file âm thanh đầy đủ: {self.output_file}")

    def run(self):
        logger.info(f"Bắt đầu xử lý file phụ đề: {self.srt_file}")
        self.generate_and_combine_audio()

def main():
    try:
        srt_file = "phude1_translated.srt"
        tts_player = SubtitleTTS(srt_file, voice="banmai", speed=1.0, output_file="full_subtitles.mp3")
        tts_player.run()
    except Exception as e:
        logger.error(f"Lỗi: {str(e)}")

if __name__ == "__main__":
    main()