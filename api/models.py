from pydantic import BaseModel, EmailStr
from typing import List, Optional

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

class UserRegister(BaseModel):
    username: str
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    email: str

class AuthResponse(BaseModel):
    success: bool
    message: str
    user: Optional[UserResponse] = None
    token: Optional[str] = None

class ModelPerformance(BaseModel):
    model_type: str
    accuracy: float
    total_predictions: int
    user_count: int