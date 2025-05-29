import sqlite3
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import json
from datetime import datetime
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from collections import defaultdict
import threading
import time

app = FastAPI(title="Video Recommendation System")

# Database setup
DB_PATH = './db.db'

def init_database():
    """Initialize database with required tables"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create interactions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            userId INTEGER NOT NULL,
            videoId INTEGER NOT NULL,
            watched_percent INTEGER CHECK(watched_percent >= 0 AND watched_percent <= 100),
            liked INTEGER CHECK(liked IN (-1, 0, 1)),
            whenReacted INTEGER CHECK(whenReacted >= 0 AND whenReacted <= 100),
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(userId, videoId)
        )
    ''')
    
    # Create labels table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS labels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            videoId INTEGER NOT NULL,
            label TEXT NOT NULL,
            FOREIGN KEY (videoId) REFERENCES videos (videoId)
        )
    ''')
    
    # Create videos table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS videos (
            videoId INTEGER PRIMARY KEY,
            text TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

# Initialize database on startup
init_database()

# Pydantic models
class Interaction(BaseModel):
    userId: int
    videoId: int
    watched_percent: Optional[int] = None
    liked: Optional[int] = 0
    whenReacted: Optional[int] = None

class VideoRecommendation(BaseModel):
    videoId: int
    score: float
    reason: str

class RecommendationResponse(BaseModel):
    recommendations: List[VideoRecommendation]
    algorithm: str

# Global variables for models
two_tower_model = None
bandit_arms = defaultdict(lambda: {'count': 0, 'reward': 0.0})
model_lock = threading.Lock()

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
        """Extract features for a video"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get video text and labels
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
        """Extract features for a user based on interaction history"""
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
        
        # Combine all interacted content with weights
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
        """Update user and video embeddings based on current data"""
        conn = sqlite3.connect(DB_PATH)
        
        # Get all users and videos
        df_interactions = pd.read_sql_query('''
            SELECT userId, videoId, watched_percent, liked
            FROM interactions
        ''', conn)
        
        df_videos = pd.read_sql_query('SELECT videoId FROM videos', conn)
        conn.close()
        
        if df_interactions.empty or df_videos.empty:
            return
        
        # Prepare text data for TF-IDF
        all_users = df_interactions['userId'].unique()
        all_videos = df_videos['videoId'].unique()
        
        # Get text representations
        video_texts = []
        user_texts = []
        
        for video_id in all_videos:
            video_texts.append(self.get_video_features(video_id))
        
        for user_id in all_users:
            user_texts.append(self.get_user_features(user_id))
        
        # Fit TF-IDF if not fitted or refit with new data
        all_texts = video_texts + user_texts
        all_texts = [text for text in all_texts if text.strip()]
        
        if all_texts:
            try:
                tfidf_matrix = self.tfidf.fit_transform(all_texts)
                
                # Split back into video and user embeddings
                video_embeddings = tfidf_matrix[:len(all_videos)]
                user_embeddings = tfidf_matrix[len(all_videos):len(all_videos) + len(all_users)]
                
                # Update embeddings dictionaries
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
        """Predict score for user-video pair"""
        if not self.is_fitted:
            self.update_embeddings()
        
        if user_id in self.user_embeddings and video_id in self.video_embeddings:
            user_emb = self.user_embeddings[user_id]
            video_emb = self.video_embeddings[video_id]
            return np.dot(user_emb, video_emb) / (np.linalg.norm(user_emb) * np.linalg.norm(video_emb) + 1e-8)
        
        return random.random() * 0.5  # Cold start fallback

class HybridRecommender:
    def __init__(self):
        self.tfidf = TfidfVectorizer(max_features=200, stop_words='english')
        self.item_similarity = None
        self.is_fitted = False
    
    def update_item_similarity(self):
        """Update item-based similarity matrix"""
        conn = sqlite3.connect(DB_PATH)
        
        # Get video content
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
            # Create TF-IDF matrix
            tfidf_matrix = self.tfidf.fit_transform(video_texts)
            
            # Calculate cosine similarity
            self.item_similarity = cosine_similarity(tfidf_matrix)
            self.video_id_to_idx = {vid: idx for idx, vid in enumerate(video_ids)}
            self.idx_to_video_id = {idx: vid for idx, vid in enumerate(video_ids)}
            self.is_fitted = True
        except Exception as e:
            print(f"Error updating item similarity: {e}")
    
    def get_recommendations(self, user_id, n_recommendations=5):
        """Get hybrid recommendations combining content and collaborative filtering"""
        if not self.is_fitted:
            self.update_item_similarity()
        
        conn = sqlite3.connect(DB_PATH)
        
        # Get user's interaction history
        cursor = conn.cursor()
        cursor.execute('''
            SELECT videoId, watched_percent, liked
            FROM interactions
            WHERE userId = ?
        ''', (user_id,))
        
        user_interactions = cursor.fetchall()
        
        # Get all available videos
        cursor.execute('SELECT videoId FROM videos')
        all_videos = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        if not user_interactions:
            # Cold start: return random videos
            random_videos = random.sample(all_videos, min(n_recommendations, len(all_videos)))
            return [(vid, random.random(), "Cold start recommendation") for vid in random_videos]
        
        # Calculate scores based on content similarity
        video_scores = {}
        interacted_videos = set()
        
        for video_id, watched_percent, liked in user_interactions:
            interacted_videos.add(video_id)
            
            # Weight based on interaction quality
            interaction_weight = (watched_percent or 0) / 100.0
            if liked == 1:
                interaction_weight *= 1.5
            elif liked == -1:
                interaction_weight *= 0.3
            
            # Find similar videos
            if video_id in self.video_id_to_idx and self.item_similarity is not None:
                video_idx = self.video_id_to_idx[video_id]
                similarities = self.item_similarity[video_idx]
                
                for idx, similarity in enumerate(similarities):
                    similar_video_id = self.idx_to_video_id[idx]
                    if similar_video_id not in interacted_videos:
                        if similar_video_id not in video_scores:
                            video_scores[similar_video_id] = 0
                        video_scores[similar_video_id] += similarity * interaction_weight
        
        # Sort and return top recommendations
        sorted_videos = sorted(video_scores.items(), key=lambda x: x[1], reverse=True)
        recommendations = []
        
        for video_id, score in sorted_videos[:n_recommendations]:
            recommendations.append((video_id, score, "Content-based similarity"))
        
        # Fill remaining slots with random videos if needed
        while len(recommendations) < n_recommendations:
            remaining_videos = [v for v in all_videos if v not in interacted_videos and v not in [r[0] for r in recommendations]]
            if not remaining_videos:
                break
            random_video = random.choice(remaining_videos)
            recommendations.append((random_video, 0.1, "Exploration"))
        
        return recommendations

class BanditRecommender:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self.arms = defaultdict(lambda: {'count': 0, 'reward': 0.0, 'avg_reward': 0.0})
    
    def update_arm(self, video_id, reward):
        """Update arm statistics after getting feedback"""
        arm = self.arms[video_id]
        arm['count'] += 1
        arm['reward'] += reward
        arm['avg_reward'] = arm['reward'] / arm['count']
    
    def get_recommendations(self, user_id, n_recommendations=5):
        """Get recommendations using epsilon-greedy bandit strategy"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get all available videos
        cursor.execute('SELECT videoId FROM videos')
        all_videos = [row[0] for row in cursor.fetchall()]
        
        # Get videos user hasn't interacted with
        cursor.execute('SELECT videoId FROM interactions WHERE userId = ?', (user_id,))
        interacted_videos = set(row[0] for row in cursor.fetchall())
        
        conn.close()
        
        available_videos = [v for v in all_videos if v not in interacted_videos]
        
        if not available_videos:
            return []
        
        recommendations = []
        
        for _ in range(min(n_recommendations, len(available_videos))):
            if random.random() < self.epsilon or not self.arms:
                # Exploration: choose random video
                video_id = random.choice(available_videos)
                reason = "Exploration (random)"
            else:
                # Exploitation: choose best performing video
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

# Initialize models
two_tower_model = TwoTowerModel()
hybrid_recommender = HybridRecommender()
bandit_recommender = BanditRecommender()

# Background model update function
def update_models():
    """Periodically update models"""
    while True:
        time.sleep(30)  # Update every 30 seconds
        with model_lock:
            try:
                two_tower_model.update_embeddings()
                hybrid_recommender.update_item_similarity()
            except Exception as e:
                print(f"Error updating models: {e}")

# Start background thread for model updates
update_thread = threading.Thread(target=update_models, daemon=True)
update_thread.start()

@app.post("/interaction")
async def save_interaction(interaction: Interaction):
    """Save user interaction to database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Insert or update interaction
        cursor.execute('''
            INSERT OR REPLACE INTO interactions 
            (userId, videoId, watched_percent, liked, whenReacted, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            interaction.userId,
            interaction.videoId,
            interaction.watched_percent,
            interaction.liked,
            interaction.whenReacted,
            datetime.now()
        ))
        
        conn.commit()
        conn.close()
        
        # Update bandit model if there's a like/dislike
        if interaction.liked != 0:
            reward = 1.0 if interaction.liked == 1 else -0.5
            bandit_recommender.update_arm(interaction.videoId, reward)
        
        # Update bandit model based on watch percentage
        if interaction.watched_percent is not None:
            watch_reward = interaction.watched_percent / 100.0
            bandit_recommender.update_arm(interaction.videoId, watch_reward)
        
        return {"message": "Interaction saved successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommend/twoTower/{user_id}")
async def recommend_two_tower(user_id: int):
    """Get recommendations using Two Tower model"""
    try:
        with model_lock:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Get all videos user hasn't interacted with
            cursor.execute('''
                SELECT videoId FROM videos 
                WHERE videoId NOT IN (
                    SELECT videoId FROM interactions WHERE userId = ?
                )
            ''', (user_id,))
            
            available_videos = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            if not available_videos:
                return RecommendationResponse(
                    recommendations=[],
                    algorithm="Two Tower"
                )
            
            # Get predictions for all available videos
            video_scores = []
            for video_id in available_videos:
                score = two_tower_model.predict(user_id, video_id)
                video_scores.append((video_id, score))
            
            # Sort by score and get top 5
            video_scores.sort(key=lambda x: x[1], reverse=True)
            top_videos = video_scores[:5]
            
            recommendations = [
                VideoRecommendation(
                    videoId=video_id,
                    score=score,
                    reason="Two Tower neural embedding similarity"
                )
                for video_id, score in top_videos
            ]
            
            return RecommendationResponse(
                recommendations=recommendations,
                algorithm="Two Tower"
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommend/hybrid/{user_id}")
async def recommend_hybrid(user_id: int):
    """Get recommendations using hybrid (item-based + content-based) approach"""
    try:
        with model_lock:
            recommendations_data = hybrid_recommender.get_recommendations(user_id, 5)
            
            recommendations = [
                VideoRecommendation(
                    videoId=video_id,
                    score=score,
                    reason=reason
                )
                for video_id, score, reason in recommendations_data
            ]
            
            return RecommendationResponse(
                recommendations=recommendations,
                algorithm="Hybrid (Item-based + Content-based)"
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommend/bandit/{user_id}")
async def recommend_bandit(user_id: int):
    """Get recommendations using Multi-Armed Bandit approach"""
    try:
        recommendations_data = bandit_recommender.get_recommendations(user_id, 5)
        
        recommendations = [
            VideoRecommendation(
                videoId=video_id,
                score=score,
                reason=reason
            )
            for video_id, score, reason in recommendations_data
        ]
        
        return RecommendationResponse(
            recommendations=recommendations,
            algorithm="Multi-Armed Bandit (Epsilon-Greedy)"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM interactions')
        interaction_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM videos')
        video_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(DISTINCT userId) FROM interactions')
        user_count = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "total_interactions": interaction_count,
            "total_videos": video_count,
            "total_users": user_count,
            "bandit_arms": len(bandit_recommender.arms),
            "two_tower_fitted": two_tower_model.is_fitted,
            "hybrid_fitted": hybrid_recommender.is_fitted
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)