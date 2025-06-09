import sqlite3
import numpy as np
import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from database import DB_PATH

class TwoTowerModel:
    def __init__(self):
        self.user_embeddings = {}
        self.video_embeddings = {}
        self.embedding_dim = 50
        self.learning_rate = 0.01
        self.scaler = StandardScaler()
        self.tfidf = TfidfVectorizer(max_features=100, stop_words='english')
        self.is_fitted = False
        
    def get_video_features(self, video_id):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT v.text, GROUP_CONCAT(l.label, ' ') as labels
            FROM videos v
            LEFT JOIN labels l ON v.videoId = l.videoId
            WHERE v.videoId = ?
            GROUP BY v.videoId
        ''', (video_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            text = result[0] or ""
            labels = result[1] or ""
            combined_text = f"{text} {labels}"
            return combined_text
        return ""
    
    def get_user_features(self, user_id):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT i.videoId, i.watched_percent, i.liked, v.text, GROUP_CONCAT(l.label, ' ') as labels
            FROM interactions i
            JOIN videos v ON i.videoId = v.videoId
            LEFT JOIN labels l ON v.videoId = l.videoId
            WHERE i.userId = ?
            GROUP BY i.videoId
        ''', (user_id,))
        
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            return ""
        
        weighted_content = []
        for video_id, watched_percent, liked, text, labels in results:
            weight = (watched_percent or 0) / 100.0
            if liked == 1:
                weight *= 2
            elif liked == -1:
                weight *= 0.5
            
            content = f"{text or ''} {labels or ''}"
            for _ in range(int(weight * 5) + 1):
                weighted_content.append(content)
        
        return " ".join(weighted_content)
    
    def update_embeddings(self):
        conn = sqlite3.connect(DB_PATH)
        
        df_interactions = pd.read_sql_query('''
            SELECT userId, videoId, watched_percent, liked
            FROM interactions
        ''', conn)
        
        df_videos = pd.read_sql_query('SELECT videoId FROM videos WHERE videoId <= 4', conn)
        conn.close()
        
        if df_interactions.empty or df_videos.empty:
            return
        
        all_users = df_interactions['userId'].unique()
        all_videos = df_videos['videoId'].unique()
        
        video_texts = []
        user_texts = []
        
        for video_id in all_videos:
            video_texts.append(self.get_video_features(video_id))
        
        for user_id in all_users:
            user_texts.append(self.get_user_features(user_id))
        
        all_texts = video_texts + user_texts
        all_texts = [text for text in all_texts if text.strip()]
        
        if all_texts:
            try:
                tfidf_matrix = self.tfidf.fit_transform(all_texts)
                
                video_embeddings = tfidf_matrix[:len(all_videos)]
                user_embeddings = tfidf_matrix[len(all_videos):len(all_videos) + len(all_users)]
                
                for i, video_id in enumerate(all_videos):
                    if i < video_embeddings.shape[0]:
                        self.video_embeddings[video_id] = video_embeddings[i].toarray().flatten()
                
                for i, user_id in enumerate(all_users):
                    if i < user_embeddings.shape[0]:
                        self.user_embeddings[user_id] = user_embeddings[i].toarray().flatten()
                
                self.is_fitted = True
            except Exception as e:
                print(f"Error updating embeddings: {e}")
    
    def predict(self, user_id, video_id):
        if not self.is_fitted:
            self.update_embeddings()
        
        if user_id in self.user_embeddings and video_id in self.video_embeddings:
            user_emb = self.user_embeddings[user_id]
            video_emb = self.video_embeddings[video_id]
            return np.dot(user_emb, video_emb) / (np.linalg.norm(user_emb) * np.linalg.norm(video_emb) + 1e-8)
        
        return random.random() * 0.5