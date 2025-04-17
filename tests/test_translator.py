import pytest
import time
import os
from pathlib import Path
from app.gemini_translate import GeminiTranslator, tasks

class TestTranslator:
    @pytest.fixture(autouse=True)
    def setup(self):
        """Khởi tạo các biến cần thiết cho test"""
        # Tạo task_id test
        self.task_id = "test_task"
        # Thêm task vào global tasks dict
        tasks[self.task_id] = {
            'status': 'processing',
            'progress': 0,
            'is_paused': False
        }
        yield
        # Cleanup sau mỗi test
        if self.task_id in tasks:
            del tasks[self.task_id]

    def test_init(self, api_key):
        """Test khởi tạo translator"""
        translator = GeminiTranslator(self.task_id)
        assert translator.task_id == self.task_id
        assert translator.api_url is not None
        assert "gemini" in translator.api_url
        assert api_key in translator.api_url

    def test_translate_text(self, api_key):
        """Test dịch văn bản"""
        translator = GeminiTranslator(self.task_id)
        
        # Test với text tiếng Trung
        chinese_text = "你好世界"
        result = translator.translate_text(chinese_text)
        assert result is not None 
        assert result != chinese_text

        # Test với text không phải tiếng Trung
        non_chinese = "Hello World"
        result = translator.translate_text(non_chinese)
        assert result == non_chinese

    def test_translate_srt(self, api_key, temp_srt_file, tmp_path):
        """Test dịch file phụ đề"""
        translator = GeminiTranslator(self.task_id)
        output_file = tmp_path / "output.srt"
        
        progress = []
        def track_progress(p):
            progress.append(p)
            
        translator.translate_srt_file(
            temp_srt_file,
            output_file,
            progress_callback=track_progress
        )
        
        # Kiểm tra file output
        assert output_file.exists()
        
        # Kiểm tra tiến độ
        assert len(progress) > 0
        assert progress[-1] == 100
        
        # Kiểm tra trạng thái task
        assert tasks[self.task_id]['status'] == 'completed'
        assert tasks[self.task_id]['progress'] == 100