import logging
from collections import defaultdict
from datetime import datetime

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global state dictionaries
tasks = {}
user_sessions = defaultdict(dict)  # Lưu trữ thông tin phiên người dùng
rate_limits = defaultdict(list)    # Lưu trữ rate limit cho từng user

def get_user_rate_limit(user_id):
    """Kiểm tra rate limit cho user cụ thể"""
    now = datetime.now()
    # Chỉ giữ lại các request trong 1 phút gần nhất
    rate_limits[user_id] = [t for t in rate_limits[user_id] if (now - t).total_seconds() < 60]
    return len(rate_limits[user_id])

def add_user_request(user_id):
    """Thêm request mới cho user"""
    rate_limits[user_id].append(datetime.now())