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
        
        cursor.execute('SELECT videoId FROM videos')
        all_videos = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        if not user_interactions:
            random_videos = random.sample(all_videos, min(n_recommendations, len(all_videos)))
            return [(vid, random.random(), "Cold start recommendation") for vid in random_videos]
        
        video_scores = {}
        interacted_videos = set()
        
        for video_id, watched_percent, liked in user_interactions:
            interacted_videos.add(video_id)
            
            interaction_weight = (watched_percent or 0) / 100.0
            if liked == 1:
                interaction_weight *= 1.5
            elif liked == -1:
                interaction_weight *= 0.3
            
            if video_id in self.video_id_to_idx and self.item_similarity is not None:
                video_idx = self.video_id_to_idx[video_id]
                similarities = self.item_similarity[video_idx]
                
                for idx, similarity in enumerate(similarities):
                    similar_video_id = self.idx_to_video_id[idx]
                    if similar_video_id not in interacted_videos:
                        if similar_video_id not in video_scores:
                            video_scores[similar_video_id] = 0
                        video_scores[similar_video_id] += similarity * interaction_weight
        
        sorted_videos = sorted(video_scores.items(), key=lambda x: x[1], reverse=True)
        recommendations = []
        
        for video_id, score in sorted_videos[:n_recommendations]:
            recommendations.append((video_id, score, "Content-based similarity"))
        
        while len(recommendations) < n_recommendations:
            remaining_videos = [v for v in all_videos if v not in interacted_videos and v not in [r[0] for r in recommendations]]
            if not remaining_videos:
                break
            random_video = random.choice(remaining_videos)
            recommendations.append((random_video, 0.1, "Exploration"))
        
        return recommendations