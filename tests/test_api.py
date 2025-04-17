import unittest
from app import app
import os
import io

class TestAPI(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.test_srt = "test.srt"
        
        # Tạo file test
        with open(self.test_srt, "w", encoding="utf-8") as f:
            f.write("""1
00:00:01,000 --> 00:00:02,000
你好世界

2
00:00:02,000 --> 00:00:03,000
很高兴见到你""")
            
    def test_translate_endpoint(self):
        """Test /translate endpoint"""
        with open(self.test_srt, 'rb') as f:
            data = {
                'file': (io.BytesIO(f.read()), 'test.srt'),
                'source_lang': 'zh',
                'target_lang': 'vi'
            }
            response = self.app.post(
                '/translate',
                content_type='multipart/form-data',
                data=data
            )
            
        self.assertEqual(response.status_code, 200)
        self.assertIn('task_id', response.json)
        
        # Test progress endpoint
        task_id = response.json['task_id']
        response = self.app.get(f'/progress/{task_id}')
        self.assertEqual(response.status_code, 200)
        self.assertIn('progress', response.json)
        
    def test_invalid_file(self):
        """Test với file không hợp lệ"""
        data = {
            'file': (io.BytesIO(b'invalid content'), 'test.txt'),
            'source_lang': 'zh'
        }
        response = self.app.post(
            '/translate',
            content_type='multipart/form-data',
            data=data
        )
        self.assertEqual(response.status_code, 400)
        
    def tearDown(self):
        # Cleanup
        if os.path.exists(self.test_srt):
            os.remove(self.test_srt)