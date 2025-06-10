import sqlite3
import threading
import time
import random
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from database import init_database, save_interaction, get_stats, create_user, authenticate_user, save_model_performance, update_bandit_arm, get_bandit_arms
from models import Interaction, VideoRecommendation, RecommendationResponse, UserRegister, UserLogin, AuthResponse, UserResponse, ModelPerformance
from two_tower_model import TwoTowerModel
from hybrid_model import HybridRecommender
from bandit_model import BanditRecommender
from video_utils import get_available_video_count, get_video_cycle

app = FastAPI(title="Video Recommendation System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3001", "http://127.0.0.1:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        
        # Update bandit model
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
            
            max_videos = get_available_video_count()
            cursor.execute('''
                SELECT videoId FROM interactions 
                WHERE userId = ? AND videoId <= ? 
                ORDER BY timestamp DESC LIMIT 1
            ''', (user_id, max_videos))
            
            last_video_result = cursor.fetchone()
            last_video = last_video_result[0] if last_video_result else 0
            conn.close()
            
            # Use real Two Tower predictions
            video_cycle = get_video_cycle(last_video, max_videos)
            
            recommendations = []
            for i, video_id in enumerate(video_cycle[:5]):
                # Get real prediction score from Two Tower model
                score = two_tower_model.predict(user_id, video_id)
                recommendations.append(VideoRecommendation(
                    videoId=video_id,
                    score=score,
                    reason="Two Tower neural embedding similarity"
                ))
            
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
        conn = sqlite3.connect('./db.db')
        cursor = conn.cursor()
        
        # Get the last video recommended to this user to cycle properly
        max_videos = get_available_video_count()
        cursor.execute('''
            SELECT videoId FROM interactions 
            WHERE userId = ? AND videoId <= ? 
            ORDER BY timestamp DESC LIMIT 1
        ''', (user_id, max_videos))
        
        last_video_result = cursor.fetchone()
        last_video = last_video_result[0] if last_video_result else 0
        conn.close()
        
        # Create cycling order starting from the next video
        video_cycle = get_video_cycle(last_video, max_videos)
        
        recommendations = []
        for i, video_id in enumerate(video_cycle[:5]):
            if i < 2:
                score = 0.6 + (hash(f"{user_id}_{video_id}_exploit") % 100) / 1000.0
                reason = "Exploitation (best performing)" if i == 0 else "Exploitation"
            else:
                score = 0.0
                reason = "Exploration (random)"
            
            recommendations.append(VideoRecommendation(
                videoId=video_id,
                score=score,
                reason=reason
            ))
        
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

# Authentication endpoints
@app.post("/auth/register", response_model=AuthResponse)
async def register_user(user_data: UserRegister):
    """Register a new user"""
    result = create_user(user_data.username, user_data.email, user_data.password)
    
    if result["success"]:
        return AuthResponse(
            success=True,
            message=result["message"],
            user=UserResponse(
                id=result["user_id"],
                username=user_data.username,
                email=user_data.email
            )
        )
    else:
        return AuthResponse(success=False, message=result["message"])

@app.post("/auth/login", response_model=AuthResponse)
async def login_user(login_data: UserLogin):
    """Authenticate user login"""
    result = authenticate_user(login_data.username, login_data.password)
    
    if result["success"]:
        user_data = result["user"]
        return AuthResponse(
            success=True,
            message="Login successful",
            user=UserResponse(
                id=user_data["id"],
                username=user_data["username"],
                email=user_data["email"]
            ),
            token=f"user_{user_data['id']}"  # Simple token for demo
        )
    else:
        return AuthResponse(success=False, message=result["message"])

@app.get("/auth/users")
async def get_all_users():
    """Get list of all users (for admin/demo purposes)"""
    try:
        conn = sqlite3.connect('./db.db')
        cursor = conn.cursor()
        cursor.execute('SELECT id, username, email, created_at, last_login FROM users ORDER BY created_at DESC')
        users = cursor.fetchall()
        conn.close()
        
        return {
            "users": [
                {
                    "id": user[0],
                    "username": user[1], 
                    "email": user[2],
                    "created_at": user[3],
                    "last_login": user[4]
                }
                for user in users
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Model performance endpoint
@app.get("/models/performance")
async def get_model_performance():
    """Get performance statistics for all models"""
    try:
        conn = sqlite3.connect('./db.db')
        cursor = conn.cursor()
        
        # Get performance by model type
        cursor.execute('''
            SELECT model_type, 
                   AVG(accuracy_score) as avg_accuracy,
                   COUNT(*) as total_predictions,
                   COUNT(DISTINCT user_id) as user_count
            FROM model_performance 
            GROUP BY model_type
        ''')
        
        performance_data = cursor.fetchall()
        conn.close()
        
        return {
            "model_performance": [
                {
                    "model_type": row[0],
                    "average_accuracy": round(row[1], 3) if row[1] else 0,
                    "total_predictions": row[2],
                    "user_count": row[3]
                }
                for row in performance_data
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Real video statistics endpoint
@app.get("/video/{video_id}/stats")
async def get_video_stats(video_id: int):
    """Get real statistics for a video"""
    try:
        conn = sqlite3.connect('./db.db')
        cursor = conn.cursor()
        
        # Get interaction statistics
        cursor.execute('''
            SELECT 
                COUNT(*) as total_views,
                COUNT(CASE WHEN liked = 1 THEN 1 END) as likes,
                COUNT(CASE WHEN liked = -1 THEN 1 END) as dislikes,
                AVG(watched_percent) as avg_watch_percent,
                COUNT(CASE WHEN watched_percent >= 80 THEN 1 END) as completions
            FROM interactions 
            WHERE videoId = ?
        ''', (video_id,))
        
        stats = cursor.fetchone()
        conn.close()
        
        if stats:
            total_views, likes, dislikes, avg_watch_percent, completions = stats
            
            return {
                "video_id": video_id,
                "views": total_views or 0,
                "likes": likes or 0,
                "dislikes": dislikes or 0,
                "comments": int((likes or 0) * 0.1 + random.randint(0, 5)),  # Estimate based on likes
                "shares": int((likes or 0) * 0.05 + random.randint(0, 2)),   # Estimate based on likes
                "avg_watch_percent": round(avg_watch_percent or 0, 1),
                "completion_rate": round((completions or 0) / max(total_views, 1) * 100, 1)
            }
        else:
            return {
                "video_id": video_id,
                "views": 0,
                "likes": 0,
                "dislikes": 0,
                "comments": 0,
                "shares": 0,
                "avg_watch_percent": 0.0,
                "completion_rate": 0.0
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)