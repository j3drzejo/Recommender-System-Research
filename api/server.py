import sqlite3
import threading
import time
from fastapi import FastAPI, HTTPException

from database import init_database, save_interaction, get_stats
from models import Interaction, VideoRecommendation, RecommendationResponse
from two_tower_model import TwoTowerModel
from hybrid_model import HybridRecommender
from bandit_model import BanditRecommender

app = FastAPI(title="Video Recommendation System")

init_database()

two_tower_model = TwoTowerModel()
hybrid_recommender = HybridRecommender()
bandit_recommender = BanditRecommender()
model_lock = threading.Lock()

def update_models():
    while True:
        time.sleep(30)
        with model_lock:
            try:
                two_tower_model.update_embeddings()
                hybrid_recommender.update_item_similarity()
            except Exception as e:
                print(f"Error updating models: {e}")

update_thread = threading.Thread(target=update_models, daemon=True)
update_thread.start()

@app.post("/interaction")
async def save_interaction_endpoint(interaction: Interaction):
    try:
        save_interaction(interaction)
        
        if interaction.liked != 0:
            reward = 1.0 if interaction.liked == 1 else -0.5
            bandit_recommender.update_arm(interaction.videoId, reward)
        
        if interaction.watched_percent is not None:
            watch_reward = interaction.watched_percent / 100.0
            bandit_recommender.update_arm(interaction.videoId, watch_reward)
        
        return {"message": "Interaction saved successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommend/twoTower/{user_id}")
async def recommend_two_tower(user_id: int):
    try:
        with model_lock:
            conn = sqlite3.connect('./db.db')
            cursor = conn.cursor()
            
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
            
            video_scores = []
            for video_id in available_videos:
                score = two_tower_model.predict(user_id, video_id)
                video_scores.append((video_id, score))
            
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
async def get_stats_endpoint():
    try:
        stats = get_stats()
        stats.update({
            "bandit_arms": len(bandit_recommender.arms),
            "two_tower_fitted": two_tower_model.is_fitted,
            "hybrid_fitted": hybrid_recommender.is_fitted
        })
        return stats
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)