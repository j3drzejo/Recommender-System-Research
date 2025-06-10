import os
import sqlite3
from database import DB_PATH

def get_available_video_count():
    # Set this to the path where your videos are stored
    video_folder = "/Users/wienio/RecommenderSystemResearch/client/public/videos/"
    
    available_videos = 0
    if os.path.exists(video_folder):
        video_files = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]
        video_numbers = []
        for f in video_files:
            try:
                num = int(f.replace('.mp4', ''))
                video_numbers.append(num)
            except ValueError:
                continue
        available_videos = max(video_numbers) if video_numbers else 0
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM videos')
    db_video_count = cursor.fetchone()[0]
    conn.close()
    
    return min(available_videos, db_video_count)

def get_video_cycle(last_video_id=None, max_videos=None):
    if max_videos is None:
        max_videos = get_available_video_count()
    
    if max_videos == 0:
        return [1, 2, 3, 4]
    
    video_cycle = list(range(1, max_videos + 1))
    
    if last_video_id and last_video_id in video_cycle:
        start_idx = (video_cycle.index(last_video_id) + 1) % len(video_cycle)
        video_cycle = video_cycle[start_idx:] + video_cycle[:start_idx]
    
    return video_cycle
