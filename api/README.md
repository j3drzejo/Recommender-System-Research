# Video Recommendation System

A FastAPI-based video recommendation system implementing three different recommendation algorithms: Two Tower Model, Hybrid Recommender, and Multi-Armed Bandit.

## Project Structure

```
.
├── server.py              # Main FastAPI application
├── database.py            # Database operations and initialization
├── models.py              # Pydantic models for API requests/responses
├── two_tower_model.py     # Two Tower neural embedding model
├── hybrid_model.py        # Hybrid recommendation model
├── bandit_model.py        # Multi-Armed Bandit model
├── requirements.txt       # Python dependencies
├── test.md               # API testing documentation
└── README.md             # This file
```

## Installation

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the server:

```bash
python server.py
```

The API will be available at `http://localhost:8000`

## Recommendation Algorithms

### 1. Two Tower Model (`/recommend/twoTower/{user_id}`)

The Two Tower architecture creates separate neural embeddings for users and videos, then computes similarity scores between them.

**How it works:**

- **User Tower**: Creates embeddings based on user's interaction history (watched videos, likes/dislikes, watch percentages)
- **Video Tower**: Creates embeddings from video content (text descriptions, labels/tags)
- **TF-IDF Vectorization**: Converts text data into numerical features
- **Cosine Similarity**: Computes similarity between user and video embeddings
- **Weighted Learning**: User embeddings are weighted by interaction quality (watch percentage, likes boost score, dislikes reduce it)

**Strengths:**

- Learns complex user preferences through neural embeddings
- Handles both content and behavioral signals
- Scalable architecture used by major platforms

**Weaknesses:**

- Requires sufficient interaction data to create meaningful embeddings
- Cold start problem for new users/videos

### 2. Hybrid Recommender (`/recommend/hybrid/{user_id}`)

Combines content-based filtering with collaborative filtering techniques using item-to-item similarity.

**How it works:**

- **Content-Based Component**: Uses TF-IDF to find videos similar to ones the user has interacted with
- **Item-Based Collaborative Filtering**: Builds similarity matrix between videos based on their features
- **Weighted Scoring**: User interactions are weighted (high watch percentage = higher weight, likes = 1.5x boost, dislikes = 0.3x penalty)
- **Exploration Component**: Adds random recommendations to avoid filter bubbles

**Strengths:**

- Works well with limited user data
- Provides interpretable recommendations
- Balances exploitation with exploration

**Weaknesses:**

- Limited by content feature quality
- May create echo chambers without sufficient exploration

### 3. Multi-Armed Bandit (`/recommend/bandit/{user_id}`)

Uses epsilon-greedy strategy to balance exploration of new content with exploitation of known good content.

**How it works:**

- **Arms**: Each video is treated as a "bandit arm" with tracked performance metrics
- **Epsilon-Greedy Strategy**: With probability ε (default 0.1), chooses random video (exploration); otherwise chooses best-performing video (exploitation)
- **Reward Learning**: Updates video performance based on user feedback (likes = +1.0 reward, dislikes = -0.5 reward, watch percentage = normalized reward)
- **Average Reward Tracking**: Maintains running average of rewards for each video

**Strengths:**

- Adapts quickly to user feedback
- Excellent for real-time learning
- Naturally handles the exploration-exploitation tradeoff

**Weaknesses:**

- Requires continuous feedback to perform well
- May be slow to converge with sparse interactions

## Database Schema

The system uses SQLite with three main tables:

- **videos**: Stores video metadata (videoId, text description)
- **labels**: Stores video tags/categories (videoId, label)
- **interactions**: Stores user interactions (userId, videoId, watched_percent, liked, whenReacted, timestamp)

## API Usage

For detailed API documentation and testing examples, see [test.md](test.md).

### Key Endpoints:

- `POST /interaction` - Record user interactions
- `GET /recommend/twoTower/{user_id}` - Get Two Tower recommendations
- `GET /recommend/hybrid/{user_id}` - Get Hybrid recommendations
- `GET /recommend/bandit/{user_id}` - Get Bandit recommendations
- `GET /stats` - Get system statistics

## Model Updates

The system includes a background thread that updates the Two Tower and Hybrid models every 30 seconds to incorporate new interaction data. The Bandit model updates immediately upon receiving feedback.

## Performance Considerations

- **Two Tower**: Most computationally expensive, best for batch processing
- **Hybrid**: Moderate complexity, good balance of accuracy and speed
- **Bandit**: Lightweight, ideal for real-time recommendations

## Cold Start Handling

- **New Users**: All models fall back to random/popular recommendations
- **New Videos**: Bandit explores new content automatically; other models use content features
- **Sparse Data**: Hybrid model includes exploration component; Two Tower uses fallback scoring
