from app.services.translator import tasks  # Import tasks from the appropriate module
import os
import logging

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def cleanup_temp_files(task_id):
    """Clean up temporary files after translation"""
    if task_id in tasks:
        task = tasks[task_id]
        try:
            # Xóa file gốc
            if os.path.exists(task['filepath']):
                os.remove(task['filepath'])
            
            # Xóa file đã dịch sau khi download
            if os.path.exists(task['output_filepath']):
                os.remove(task['output_filepath'])
                
            # Xóa task data
            del tasks[task_id]
            
        except Exception as e:
            logger.error(f"Failed to cleanup files: {str(e)}")