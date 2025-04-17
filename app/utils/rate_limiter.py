from datetime import datetime
import time

class RateLimiter:
    def __init__(self, requests_per_min=12):
        self.requests_per_min = requests_per_min
        self.requests = []
        
    def wait_if_needed(self):
        now = datetime.now()
        self.requests = [r for r in self.requests if (now - r).seconds < 60]
        if len(self.requests) >= self.requests_per_min:
            sleep_time = 60 - (now - self.requests[0]).seconds
            if sleep_time > 0:
                time.sleep(sleep_time)
        self.requests.append(now)

from functools import lru_cache

@lru_cache(maxsize=1000)
def translate_cached_text(text, source_lang, target_lang):
    # Cache kết quả dịch để tái sử dụng
    # Placeholder implementation for translate_text
    def translate_text(text, source_lang, target_lang):
        # Replace this with the actual translation logic
        return f"Translated '{text}' from {source_lang} to {target_lang}"
    
    return translate_text(text, source_lang, target_lang)