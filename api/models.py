from pydantic import BaseModel
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