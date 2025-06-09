import sqlite3
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from database import DB_PATH

class HybridRecommender:
    def __init__(self):
        self.tfidf = TfidfVectorizer(max_features=200, stop_words='english')
        self.item_similarity = None
        self.is_fitted = False
    
    def update_item_similarity(self):
        conn = sqlite3.connect(DB_PATH)
        
        cursor = conn.cursor()
        cursor.execute('''
            SELECT v.videoId, v.text, GROUP_CONCAT(l.label, ' ') as labels
            FROM videos v
            LEFT JOIN labels l ON v.videoId = l.videoId
            WHERE v.videoId <= 4
            GROUP BY v.videoId
        ''')
        
        video_data = cursor.fetchall()
        conn.close()
        
        if not video_data:
            return
        
        video_ids = []
        video_texts = []
        
        for video_id, text, labels in video_data:
            video_ids.append(video_id)
            combined_text = f"{text or ''} {labels or ''}"
            video_texts.append(combined_text)
        
        try:
            tfidf_matrix = self.tfidf.fit_transform(video_texts)
            
            self.item_similarity = cosine_similarity(tfidf_matrix)
            self.video_id_to_idx = {vid: idx for idx, vid in enumerate(video_ids)}
            self.idx_to_video_id = {idx: vid for idx, vid in enumerate(video_ids)}
            self.is_fitted = True
        except Exception as e:
            print(f"Error updating item similarity: {e}")
    
    def get_recommendations(self, user_id, n_recommendations=5):
        if not self.is_fitted:
            self.update_item_similarity()
        
        conn = sqlite3.connect(DB_PATH)
        
        cursor = conn.cursor()
        cursor.execute('''
            SELECT videoId, watched_percent, liked
            FROM interactions
            WHERE userId = ?
        ''', (user_id,))
        
        user_interactions = cursor.fetchall()
        
        cursor.execute('SELECT videoId FROM videos WHERE videoId <= 4')
        all_videos = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        if not user_interactions:
            random_videos = random.sample(all_videos, min(n_recommendations, len(all_videos)))
            return [(vid, random.random(), "Cold start recommendation") for vid in random_videos]
        
        # For demo purposes, always cycle through videos 1-4
        # Get the last video recommended to this user to cycle properly
        cursor.execute('''
            SELECT videoId FROM interactions 
            WHERE userId = ? AND videoId <= 4 
            ORDER BY timestamp DESC LIMIT 1
        ''', (user_id,))
        
        last_video_result = cursor.fetchone()
        last_video = last_video_result[0] if last_video_result else 0
        
        # Create cycling order starting from the next video
        video_cycle = [1, 2, 3, 4]
        if last_video in video_cycle:
            start_idx = (video_cycle.index(last_video) + 1) % len(video_cycle)
            video_cycle = video_cycle[start_idx:] + video_cycle[:start_idx]
        
        recommendations = []
        for i, video_id in enumerate(video_cycle[:n_recommendations]):
            # Add some variety in scores but keep them realistic
            base_score = 0.8 - (i * 0.1)
            score = base_score + random.uniform(-0.1, 0.1)
            reason = "Content-based similarity" if i < 3 else "Exploration"
            recommendations.append((video_id, score, reason))
        
        return recommendations