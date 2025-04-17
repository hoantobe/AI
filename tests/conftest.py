import pytest
import os
from dotenv import load_dotenv

@pytest.fixture(scope="session", autouse=True)
def load_env():
    load_dotenv()

@pytest.fixture(scope="session")
def api_key():
    return os.getenv("GEMINI_API_KEY")

@pytest.fixture
def test_srt_content():
    return """1
00:00:01,000 --> 00:00:02,000
你好世界

2
00:00:02,000 --> 00:00:03,000
很高兴见到你"""

@pytest.fixture
def temp_srt_file(tmp_path, test_srt_content):
    srt_file = tmp_path / "test.srt"
    srt_file.write_text(test_srt_content, encoding="utf-8")
    return srt_file