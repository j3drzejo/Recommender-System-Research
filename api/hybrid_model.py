import sqlite3
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from database import DB_PATH
from video_utils import get_available_video_count

class HybridRecommender:
    def __init__(self):
        self.tfidf = TfidfVectorizer(max_features=200, stop_words='english')
        self.item_similarity = None
        self.is_fitted = False
        self.video_id_to_idx = {}
        self.idx_to_video_id = {}
    
    def update_item_similarity(self):
        """Build TF-IDF similarity matrix for content-based filtering"""
        conn = sqlite3.connect(DB_PATH)
        
        cursor = conn.cursor()
        max_videos = get_available_video_count()
        cursor.execute('''
            SELECT v.videoId, v.text, GROUP_CONCAT(l.label, ' ') as labels
            FROM videos v
            LEFT JOIN labels l ON v.videoId = l.videoId
            WHERE v.videoId <= ?
            GROUP BY v.videoId
        ''', (max_videos,))
        
        video_data = cursor.fetchall()
        conn.close()
        
        if not video_data:
            print("No video data found for similarity calculation")
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
            
            # Calculate cosine similarity matrix
            self.item_similarity = cosine_similarity(tfidf_matrix)
            self.video_id_to_idx = {vid: idx for idx, vid in enumerate(video_ids)}
            self.idx_to_video_id = {idx: vid for idx, vid in enumerate(video_ids)}
            self.is_fitted = True
            print(f"Updated item similarity matrix for {len(video_ids)} videos")
        except Exception as e:
            print(f"Error updating item similarity: {e}")
    
    def get_recommendations(self, user_id, n_recommendations=5):
        """Generate hybrid recommendations using content-based + collaborative filtering"""
        if not self.is_fitted:
            self.update_item_similarity()

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get user interaction history
        cursor.execute('''
            SELECT videoId, watched_percent, liked
            FROM interactions
            WHERE userId = ?
        ''', (user_id,))
        
        user_interactions = cursor.fetchall()
        
        max_videos = get_available_video_count()
        cursor.execute('SELECT videoId FROM videos WHERE videoId <= ?', (max_videos,))
        all_videos = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        
        # If no interactions, use content-based recommendations for cold start
        if not user_interactions:
            return self._cold_start_recommendations(all_videos[:n_recommendations])
        
        # REAL HYBRID ALGORITHM: Content-based + Collaborative Filtering
        video_scores = {}
        interacted_videos = set()
        
        # Process each user interaction to build preferences
        for video_id, watched_percent, liked in user_interactions:
            interacted_videos.add(video_id)
            
            # Calculate interaction strength (0.0 to 1.0)
            interaction_weight = (watched_percent or 0) / 100.0
            
            # Apply like/dislike multipliers
            if liked == 1:
                interaction_weight *= 1.5  # Boost liked content
            elif liked == -1:
                interaction_weight *= 0.2  # Reduce disliked content weight
            
            # Skip if user disliked this heavily
            if liked == -1 and watched_percent and watched_percent < 20:
                continue
                
            # Find similar videos using TF-IDF cosine similarity
            if video_id in self.video_id_to_idx and self.item_similarity is not None:
                video_idx = self.video_id_to_idx[video_id]
                similarities = self.item_similarity[video_idx]
                
                # Calculate similarity scores for all other videos
                for idx, similarity in enumerate(similarities):
                    similar_video_id = self.idx_to_video_id[idx]
                    
                    # Don't recommend already seen videos
                    if similar_video_id in interacted_videos:
                        continue
                        
                    # Only recommend available videos
                    if similar_video_id not in all_videos:
                        continue
                    
                    # Weighted similarity score
                    weighted_score = similarity * interaction_weight
                    
                    if similar_video_id not in video_scores:
                        video_scores[similar_video_id] = {
                            'content_score': 0.0,
                            'interaction_count': 0,
                            'total_weight': 0.0
                        }
                    
                    video_scores[similar_video_id]['content_score'] += weighted_score
                    video_scores[similar_video_id]['interaction_count'] += 1
                    video_scores[similar_video_id]['total_weight'] += interaction_weight
        
        # Calculate final scores and rank videos
        final_scores = []
        for video_id, scores in video_scores.items():
            # Normalize by number of interactions that contributed
            if scores['interaction_count'] > 0:
                # Average content similarity weighted by interaction strength
                content_score = scores['content_score'] / scores['interaction_count']
                
                # Boost videos that multiple interactions point to
                popularity_boost = min(scores['interaction_count'] * 0.1, 0.3)
                
                # Final hybrid score
                final_score = content_score + popularity_boost
                
                reason = f"Content similarity ({content_score:.3f})"
                if popularity_boost > 0:
                    reason += f" + multi-interaction boost"
                    
                final_scores.append((video_id, final_score, reason))
        
        # Sort by score descending
        final_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Add exploration if we don't have enough recommendations
        recommendations = final_scores[:n_recommendations]
        
        # Fill remaining slots with exploration (unseen videos)
        if len(recommendations) < n_recommendations:
            unseen_videos = [v for v in all_videos 
                           if v not in interacted_videos 
                           and v not in [r[0] for r in recommendations]]
            
            # Add exploration videos with diversity-based scores
            for video_id in unseen_videos[:n_recommendations - len(recommendations)]:
                exploration_score = random.uniform(0.1, 0.3)
                recommendations.append((video_id, exploration_score, "Exploration (unseen content)"))
        
        return recommendations
    
    def _cold_start_recommendations(self, video_ids):
        """Provide recommendations for new users with no interaction history"""
        recommendations = []
        for i, video_id in enumerate(video_ids):
            # Use content diversity as initial score
            score = 0.6 - (i * 0.05)  # Gradually decreasing scores
            reason = "Cold start (content-based)"
            recommendations.append((video_id, score, reason))
        return recommendations
    
    def get_content_similarity(self, video_id1, video_id2):
        """Get similarity score between two videos"""
        if not self.is_fitted or video_id1 not in self.video_id_to_idx or video_id2 not in self.video_id_to_idx:
            return 0.0
        
        idx1 = self.video_id_to_idx[video_id1]
        idx2 = self.video_id_to_idx[video_id2]
        
        return float(self.item_similarity[idx1][idx2])
