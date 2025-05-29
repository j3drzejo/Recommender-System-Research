# Video Recommendation System API

## Base URL

http://0.0.0.0:8000

---

## POST `/interaction`

Save a user interaction (watch, like, dislike, etc.)

### Request Body

```json
{
  "userId": 1,
  "videoId": 101,
  "watched_percent": 80,
  "liked": 1,
  "whenReacted": 60
}

    liked: 1 (like), -1 (dislike), 0 (neutral)

    watched_percent: Optional integer between 0 and 100

    whenReacted: Optional integer between 0 and 100

Response

{
  "message": "Interaction saved successfully"
}

GET /recommend/twoTower/{user_id}

Get 5 recommendations for a user using the Two Tower model.
Example

GET /recommend/twoTower/1

Response

{
  "recommendations": [
    { "videoId": 105, "score": 0.8123, "reason": "Two Tower neural embedding similarity" },
    { "videoId": 110, "score": 0.7791, "reason": "Two Tower neural embedding similarity" },
    { "videoId": 108, "score": 0.7564, "reason": "Two Tower neural embedding similarity" },
    { "videoId": 107, "score": 0.7321, "reason": "Two Tower neural embedding similarity" },
    { "videoId": 112, "score": 0.7105, "reason": "Two Tower neural embedding similarity" }
  ],
  "algorithm": "Two Tower"
}

GET /recommend/hybrid/{user_id}

Get 5 recommendations using a Hybrid model (item-based + content-based).
Example

GET /recommend/hybrid/1

Response

{
  "recommendations": [
    { "videoId": 103, "score": 0.745, "reason": "Content-based similarity" },
    { "videoId": 107, "score": 0.705, "reason": "Item-based similarity" },
    { "videoId": 114, "score": 0.695, "reason": "Content-based similarity" },
    { "videoId": 120, "score": 0.682, "reason": "Item-based similarity" },
    { "videoId": 130, "score": 0.1, "reason": "Exploration" }
  ],
  "algorithm": "Hybrid (Item-based + Content-based)"
}

GET /recommend/bandit/{user_id}

Get 5 recommendations using a Multi-Armed Bandit (epsilon-greedy) strategy.
Example

GET /recommend/bandit/1

Response

{
  "recommendations": [
    { "videoId": 104, "score": 0.667, "reason": "Exploitation (best performing)" },
    { "videoId": 106, "score": 0.625, "reason": "Exploitation" },
    { "videoId": 110, "score": 0.6, "reason": "Exploitation" },
    { "videoId": 121, "score": 0.0, "reason": "Exploration (random)" },
    { "videoId": 122, "score": 0.0, "reason": "Exploration (random)" }
  ],
  "algorithm": "Multi-Armed Bandit (Epsilon-Greedy)"
}

GET /stats

Get general statistics about the system.
Example

GET /stats

Response

{
  "total_interactions": 120,
  "total_videos": 50,
  "total_users": 10,
  "bandit_arms": 40,
  "two_tower_fitted": true,
  "hybrid_fitted": true
}
```
