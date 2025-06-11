import sqlite3
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from database import DB_PATH
from video_utils import get_available_video_count

class TwoTowerModel:
    def __init__(self):
        self.user_embeddings = {}
        self.video_embeddings = {}
        self.embedding_dim = 64
        self.tfidf_vectorizer = TfidfVectorizer(max_features=200, stop_words='english')
        self.user_scaler = StandardScaler()
        self.video_scaler = StandardScaler()
        self.svd_user = TruncatedSVD(n_components=self.embedding_dim, random_state=42)
        self.svd_video = TruncatedSVD(n_components=self.embedding_dim, random_state=42)
        self.is_fitted = False
        
    def get_video_features(self, video_id):
        """Extract text features for a video"""
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
    
    def get_user_interaction_features(self, user_id):
        """Extract user behavior features from interaction history"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT i.videoId, i.watched_percent, i.liked, v.text, 
                   GROUP_CONCAT(l.label, ' ') as labels
            FROM interactions i
            JOIN videos v ON i.videoId = v.videoId
            LEFT JOIN labels l ON v.videoId = l.videoId
            WHERE i.userId = ?
            GROUP BY i.videoId
        ''', (user_id,))
        
        interactions = cursor.fetchall()
        conn.close()
        
        if not interactions:
            return {}
        
        # Calculate user preference features
        features = {
            'total_interactions': len(interactions),
            'avg_watch_percent': np.mean([i[1] or 0 for i in interactions]),
            'like_ratio': len([i for i in interactions if i[2] == 1]) / len(interactions),
            'dislike_ratio': len([i for i in interactions if i[2] == -1]) / len(interactions),
            'completion_rate': len([i for i in interactions if (i[1] or 0) > 80]) / len(interactions),
            'engagement_score': 0.0
        }
        
        # Calculate engagement score based on watch time and likes
        total_engagement = 0
        for video_id, watch_percent, liked, text, labels in interactions:
            engagement = (watch_percent or 0) / 100.0
            if liked == 1:
                engagement *= 1.5
            elif liked == -1:
                engagement *= 0.5
            total_engagement += engagement
        
        features['engagement_score'] = total_engagement / len(interactions)
        
        # Aggregate content preferences (TF-IDF of interacted content)
        user_content = []
        for _, _, _, text, labels in interactions:
            if text:
                user_content.append(f"{text} {labels or ''}")
        
        features['content_text'] = ' '.join(user_content)
        return features
    
    def update_embeddings(self):
        """Build user and video embeddings using matrix factorization"""
        conn = sqlite3.connect(DB_PATH)
        
        # Get all interactions
        df_interactions = pd.read_sql_query('''
            SELECT userId, videoId, watched_percent, liked
            FROM interactions
        ''', conn)
        
        max_videos = get_available_video_count()
        df_videos = pd.read_sql_query(f'''
            SELECT videoId, text FROM videos WHERE videoId <= {max_videos}
        ''', conn)
        
        conn.close()
        
        if df_interactions.empty or df_videos.empty:
            print("No data available for Two Tower training")
            return
        
        print(f"Training Two Tower model with {len(df_interactions)} interactions and {len(df_videos)} videos")
        
        # Build user-item interaction matrix
        all_users = df_interactions['userId'].unique()
        all_videos = df_videos['videoId'].unique()
        
        # Create user features matrix
        user_features = []
        user_ids = []
        
        for user_id in all_users:
            user_data = self.get_user_interaction_features(user_id)
            if user_data:
                # Numerical features
                feature_vector = [
                    user_data['total_interactions'],
                    user_data['avg_watch_percent'],
                    user_data['like_ratio'],
                    user_data['dislike_ratio'],
                    user_data['completion_rate'],
                    user_data['engagement_score']
                ]
                user_features.append(feature_vector)
                user_ids.append(user_id)
        
        # Create video features matrix
        video_features = []
        video_ids = []
        video_texts = []
        
        for video_id in all_videos:
            video_text = self.get_video_features(video_id)
            if video_text.strip():
                video_texts.append(video_text)
                video_ids.append(video_id)
        
        if not user_features or not video_texts:
            print("Insufficient feature data for Two Tower training")
            return
        
        try:
            # Process video text features with TF-IDF
            video_tfidf = self.tfidf_vectorizer.fit_transform(video_texts)
            
            # Normalize user features
            user_features = np.array(user_features)
            user_features_scaled = self.user_scaler.fit_transform(user_features)
            
            # Create embeddings using SVD (dimensionality reduction)
            # For users: use their behavioral features
            if user_features_scaled.shape[1] < self.embedding_dim:
                # Pad with zeros if we have fewer features than embedding dimensions
                padding = np.zeros((user_features_scaled.shape[0], 
                                  self.embedding_dim - user_features_scaled.shape[1]))
                user_features_padded = np.hstack([user_features_scaled, padding])
                user_embeddings = user_features_padded
            else:
                user_embeddings = self.svd_user.fit_transform(user_features_scaled)
            
            # For videos: use TF-IDF features
            video_embeddings = self.svd_video.fit_transform(video_tfidf)
            
            # Store embeddings
            self.user_embeddings = {user_ids[i]: user_embeddings[i] 
                                  for i in range(len(user_ids))}
            self.video_embeddings = {video_ids[i]: video_embeddings[i] 
                                   for i in range(len(video_ids))}
            
            self.is_fitted = True
            print(f"Two Tower model trained successfully:")
            print(f"  - User embeddings: {len(self.user_embeddings)} users x {self.embedding_dim} dims")
            print(f"  - Video embeddings: {len(self.video_embeddings)} videos x {self.embedding_dim} dims")
            
        except Exception as e:
            print(f"Error training Two Tower model: {e}")
    
    def predict(self, user_id, video_id):
        """Predict user-video interaction score using embedding similarity"""
        if not self.is_fitted:
            self.update_embeddings()
        
        # Get embeddings
        user_embedding = self.user_embeddings.get(user_id)
        video_embedding = self.video_embeddings.get(video_id)
        
        if user_embedding is None or video_embedding is None:
            # Cold start: use content-based fallback
            return self._cold_start_prediction(user_id, video_id)
        
        # Calculate cosine similarity between user and video embeddings
        user_vec = user_embedding.reshape(1, -1)
        video_vec = video_embedding.reshape(1, -1)
        
        similarity = cosine_similarity(user_vec, video_vec)[0][0]
        
        # Convert similarity (-1 to 1) to prediction score (0 to 1)
        score = (similarity + 1) / 2
        
        return max(0.0, min(1.0, score))
    
    def _cold_start_prediction(self, user_id, video_id):
        """Fallback prediction for new users/videos"""
        # Use average score with some randomness
        base_score = 0.5
        
        # Check if user has any interactions to bias the prediction
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT AVG(CASE 
                WHEN liked = 1 THEN 0.8
                WHEN liked = -1 THEN 0.2
                ELSE 0.5
            END) as avg_preference
            FROM interactions 
            WHERE userId = ?
        ''', (user_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result and result[0] is not None:
            base_score = float(result[0])
        
        # Add small random variation
        score = base_score + np.random.normal(0, 0.1)
        return max(0.0, min(1.0, score))
    
    def get_user_embedding(self, user_id):
        """Get user embedding vector"""
        return self.user_embeddings.get(user_id)
    
    def get_video_embedding(self, video_id):
        """Get video embedding vector"""
        return self.video_embeddings.get(video_id)
    
    def get_model_stats(self):
        """Get model statistics"""
        return {
            'is_fitted': self.is_fitted,
            'num_users': len(self.user_embeddings),
            'num_videos': len(self.video_embeddings),
            'embedding_dim': self.embedding_dim
        }
