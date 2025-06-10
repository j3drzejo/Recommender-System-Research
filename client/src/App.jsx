import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import {
  Heart,
  MessageCircle,
  Share,
  ThumbsDown,
  RotateCcw,
  User,
  LogOut,
} from "lucide-react";
import axios from "axios";
import AuthComponent from "./AuthComponent";
import "./App.css";

const CURRENT_USER_ID = 1;
const API_BASE_URL = "http://localhost:8000";

function App() {
  const [user, setUser] = useState(null);
  const [videos, setVideos] = useState([]);
  const [currentVideoIndex, setCurrentVideoIndex] = useState(0);
  const [loading, setLoading] = useState(false);
  const [selectedAlgorithm, setSelectedAlgorithm] = useState("hybrid");
  const [userInteractions, setUserInteractions] = useState({});
  const [videoProgress, setVideoProgress] = useState(0);
  const videoRef = useRef(null);

  // Check for existing user session on component mount
  useEffect(() => {
    const savedUser = localStorage.getItem("user");
    if (savedUser) {
      try {
        setUser(JSON.parse(savedUser));
      } catch (error) {
        console.error("Error parsing saved user:", error);
        localStorage.removeItem("user");
        localStorage.removeItem("token");
      }
    }
  }, []);

  // Create stable axios instance
  const api = useMemo(
    () =>
      axios.create({
        baseURL: API_BASE_URL,
        timeout: 10000,
      }),
    []
  );

  const fetchRecommendations = useCallback(
    async (algorithm = "hybrid") => {
      setLoading(true);
      try {
        const userId = user?.id || CURRENT_USER_ID;
        const response = await api.get(`/recommend/${algorithm}/${userId}`);
        const recommendations = response.data.recommendations;

        // Fetch real statistics for each video
        const videoObjects = await Promise.all(
          recommendations.map(async (rec) => {
            try {
              const statsResponse = await api.get(
                `/video/${rec.videoId}/stats`
              );
              const stats = statsResponse.data;

              return {
                id: rec.videoId,
                title: `Story ${rec.videoId}`,
                score: rec.score,
                reason: rec.reason,
                algorithm: response.data.algorithm,
                videoUrl: `/videos/${rec.videoId}.mp4`,
                author: "StoryBot",
                likes: stats.likes,
                comments: stats.comments,
                shares: stats.shares,
                views: stats.views,
                avgWatchPercent: stats.avg_watch_percent,
                completionRate: stats.completion_rate,
              };
            } catch (error) {
              console.error(
                `Error fetching stats for video ${rec.videoId}:`,
                error
              );
              // Fallback to basic data if stats fetch fails
              return {
                id: rec.videoId,
                title: `Story ${rec.videoId}`,
                score: rec.score,
                reason: rec.reason,
                algorithm: response.data.algorithm,
                videoUrl: `/videos/${rec.videoId}.mp4`,
                author: "StoryBot",
                likes: 0,
                comments: 0,
                shares: 0,
                views: 0,
                avgWatchPercent: 0,
                completionRate: 0,
              };
            }
          })
        );

        setVideos((prevVideos) => [...prevVideos, ...videoObjects]);
      } catch (error) {
        console.error("Error fetching recommendations:", error);
      } finally {
        setLoading(false);
      }
    },
    [api, user]
  );

  const sendInteraction = useCallback(
    async (videoId, liked, watchedPercent) => {
      try {
        const userId = user?.id || CURRENT_USER_ID;
        await api.post("/interaction", {
          userId: userId,
          videoId: videoId,
          watched_percent: Math.round(watchedPercent),
          liked: liked,
          whenReacted: Math.round(watchedPercent),
        });

        setUserInteractions((prev) => ({
          ...prev,
          [videoId]: { liked, watchedPercent },
        }));
      } catch (error) {
        console.error("Error sending interaction:", error);
      }
    },
    [api, user]
  );

  // Initial fetch when user logs in or algorithm changes
  useEffect(() => {
    if (user) {
      setVideos([]);
      setCurrentVideoIndex(0);
      setUserInteractions({});
      fetchRecommendations(selectedAlgorithm);
    }
  }, [selectedAlgorithm, user, fetchRecommendations]);

  // Load more videos when running low (but not on initial load)
  useEffect(() => {
    const remainingVideos = videos.length - currentVideoIndex;
    if (
      remainingVideos <= 2 &&
      remainingVideos > 0 &&
      !loading &&
      videos.length > 0 &&
      user
    ) {
      fetchRecommendations(selectedAlgorithm);
    }
  }, [
    currentVideoIndex,
    videos.length,
    selectedAlgorithm,
    loading,
    user,
    fetchRecommendations,
  ]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const updateProgress = () => {
      const progress = (video.currentTime / video.duration) * 100;
      setVideoProgress(progress);
    };

    video.addEventListener("timeupdate", updateProgress);
    return () => video.removeEventListener("timeupdate", updateProgress);
  }, [currentVideoIndex]);

  const handleLike = () => {
    const currentVideo = videos[currentVideoIndex];
    if (!currentVideo) return;

    const isLiked = userInteractions[currentVideo.id]?.liked === 1;
    const newLiked = isLiked ? 0 : 1;

    sendInteraction(currentVideo.id, newLiked, videoProgress);
  };

  const handleDislike = () => {
    const currentVideo = videos[currentVideoIndex];
    if (!currentVideo) return;

    const isDisliked = userInteractions[currentVideo.id]?.liked === -1;
    const newLiked = isDisliked ? 0 : -1;

    sendInteraction(currentVideo.id, newLiked, videoProgress);

    setTimeout(() => {
      nextVideo();
    }, 500);
  };

  const nextVideo = () => {
    if (currentVideoIndex < videos.length - 1) {
      setCurrentVideoIndex((prev) => prev + 1);
    } else {
      fetchRecommendations(selectedAlgorithm);
    }
  };

  const prevVideo = () => {
    if (currentVideoIndex > 0) {
      setCurrentVideoIndex((prev) => prev - 1);
    }
  };

  const refreshFeed = () => {
    setVideos([]);
    setCurrentVideoIndex(0);
    fetchRecommendations(selectedAlgorithm);
  };

  const handleLogin = (userData) => {
    setUser(userData);
  };

  const handleLogout = () => {
    setUser(null);
    setVideos([]);
    setCurrentVideoIndex(0);
    setUserInteractions({});
    localStorage.removeItem("user");
    localStorage.removeItem("token");
  };

  // If no user is logged in, show authentication
  if (!user) {
    return <AuthComponent onLogin={handleLogin} />;
  }

  const currentVideo = videos[currentVideoIndex];
  const currentInteraction = currentVideo
    ? userInteractions[currentVideo.id]
    : null;

  return (
    <div className="app">
      <header className="app-header">
        <div className="algorithm-selector">
          <select
            value={selectedAlgorithm}
            onChange={(e) => setSelectedAlgorithm(e.target.value)}
            className="algorithm-select"
          >
            <option value="hybrid">Hybrid</option>
            <option value="twoTower">Two Tower</option>
            <option value="bandit">Bandit</option>
          </select>
        </div>
        <h1>StoryFeed</h1>
        <div className="header-actions">
          <span className="username">@{user.username}</span>
          <button onClick={handleLogout} className="logout-btn">
            <LogOut size={16} />
          </button>
          <button onClick={refreshFeed} className="refresh-btn">
            <RotateCcw size={20} />
          </button>
        </div>
      </header>

      <div className="video-container">
        {currentVideo ? (
          <>
            <div className="video-player">
              <video
                ref={videoRef}
                src={currentVideo.videoUrl}
                autoPlay
                loop
                muted
                playsInline
                onError={() =>
                  console.log("Video failed to load:", currentVideo.videoUrl)
                }
              />
              <div className="video-info">
                <div className="video-meta">
                  <h3>{currentVideo.title}</h3>
                  <p>@{currentVideo.author}</p>
                  <div className="algorithm-info">
                    <span className="algorithm-tag">
                      {currentVideo.algorithm}
                    </span>
                    <span className="score">
                      Score: {currentVideo.score.toFixed(3)}
                    </span>
                  </div>
                  <p className="reason">{currentVideo.reason}</p>
                  <div className="video-stats">
                    <span>👁️ {currentVideo.views} views</span>
                    <span>
                      📊 {currentVideo.avgWatchPercent?.toFixed(1)}% avg watch
                    </span>
                    <span>
                      ✅ {currentVideo.completionRate?.toFixed(1)}% completion
                    </span>
                  </div>
                </div>
              </div>
              <div className="progress-bar">
                <div
                  className="progress-fill"
                  style={{ width: `${videoProgress}%` }}
                />
              </div>
            </div>
            <div className="interactions">
              <button
                className={`interaction-btn ${
                  currentInteraction?.liked === 1 ? "active" : ""
                }`}
                onClick={handleLike}
              >
                <Heart
                  size={24}
                  fill={currentInteraction?.liked === 1 ? "#ff0050" : "none"}
                />
                <span>
                  {currentVideo.likes +
                    (currentInteraction?.liked === 1 ? 1 : 0)}
                </span>
              </button>

              <button
                className={`interaction-btn ${
                  currentInteraction?.liked === -1 ? "active" : ""
                }`}
                onClick={handleDislike}
              >
                <ThumbsDown
                  size={24}
                  fill={currentInteraction?.liked === -1 ? "#666" : "none"}
                />
              </button>

              <button className="interaction-btn">
                <MessageCircle size={24} />
                <span>{currentVideo.comments}</span>
              </button>

              <button className="interaction-btn">
                <Share size={24} />
                <span>{currentVideo.shares}</span>
              </button>

              <div className="user-avatar">
                <User size={24} />
              </div>
            </div>
          </>
        ) : (
          <div className="loading">
            {loading ? "Loading recommendations..." : "No videos available"}
          </div>
        )}
      </div>
      <div className="navigation">
        <button
          onClick={prevVideo}
          disabled={currentVideoIndex === 0}
          className="nav-btn"
        >
          ↑ Previous
        </button>
        <span className="video-counter">
          {currentVideoIndex + 1} / {videos.length}
        </span>
        <button onClick={nextVideo} className="nav-btn">
          ↓ Next
        </button>
      </div>
      <div className="debug-info">
        <p>
          User: {user.username} (ID: {user.id})
        </p>
        <p>Algorithm: {selectedAlgorithm}</p>
        <p>Progress: {Math.round(videoProgress)}%</p>
        {currentInteraction && (
          <p>
            Interaction:{" "}
            {currentInteraction.liked === 1
              ? "Liked"
              : currentInteraction.liked === -1
              ? "Disliked"
              : "Neutral"}
          </p>
        )}
      </div>
    </div>
  );
}

export default App;
