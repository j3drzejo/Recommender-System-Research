import sqlite3
import random
from collections import defaultdict
from database import DB_PATH

class BanditRecommender:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self.arms = defaultdict(lambda: {'count': 0, 'reward': 0.0, 'avg_reward': 0.0})
    
    def update_arm(self, video_id, reward):
        arm = self.arms[video_id]
        arm['count'] += 1
        arm['reward'] += reward
        arm['avg_reward'] = arm['reward'] / arm['count']
    
    def get_recommendations(self, user_id, n_recommendations=5):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('SELECT videoId FROM videos')
        all_videos = [row[0] for row in cursor.fetchall()]
        
        cursor.execute('SELECT videoId FROM interactions WHERE userId = ?', (user_id,))
        interacted_videos = set(row[0] for row in cursor.fetchall())
        
        conn.close()
        
        available_videos = [v for v in all_videos if v not in interacted_videos]
        
        if not available_videos:
            return []
        
        recommendations = []
        
        for _ in range(min(n_recommendations, len(available_videos))):
            if random.random() < self.epsilon or not self.arms:
                video_id = random.choice(available_videos)
                reason = "Exploration (random)"
            else:
                best_videos = [(vid, self.arms[vid]['avg_reward']) for vid in available_videos if vid in self.arms]
                if best_videos:
                    video_id = max(best_videos, key=lambda x: x[1])[0]
                    reason = "Exploitation (best performing)"
                else:
                    video_id = random.choice(available_videos)
                    reason = "Cold start (no data)"
            
            score = self.arms[video_id]['avg_reward'] if video_id in self.arms else 0.0
            recommendations.append((video_id, score, reason))
            available_videos.remove(video_id)
        
        return recommendations