import { useState } from "react";
import { Eye, EyeOff, User, Mail, Lock, Smartphone } from "lucide-react";
import axios from "axios";
import "./Auth.css";

const API_BASE_URL = "http://localhost:8000";

function AuthComponent({ onLogin }) {
  const [isLogin, setIsLogin] = useState(true);
  const [formData, setFormData] = useState({
    username: "",
    email: "",
    password: "",
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [showPassword, setShowPassword] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");

    try {
      const endpoint = isLogin ? "/auth/login" : "/auth/register";
      const data = isLogin
        ? { username: formData.username, password: formData.password }
        : formData;

      const response = await axios.post(`${API_BASE_URL}${endpoint}`, data);

      if (response.data.success) {
        const user = response.data.user;
        localStorage.setItem("user", JSON.stringify(user));
        localStorage.setItem("token", response.data.token || `user_${user.id}`);
        onLogin(user);
      } else {
        setError(response.data.message);
      }
    } catch (error) {
      setError(error.response?.data?.detail || "An error occurred");
    } finally {
      setLoading(false);
    }
  };

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  const demoLogin = async () => {
    try {
      // Create a demo user if not exists, then login
      await axios
        .post(`${API_BASE_URL}/auth/register`, {
          username: "demo_user",
          email: "demo@example.com",
          password: "demo123",
        })
        .catch(() => {}); // Ignore if user already exists

      setFormData({ username: "demo_user", password: "demo123" });
      setIsLogin(true);

      // Auto-submit
      setTimeout(() => {
        document.querySelector("form").requestSubmit();
      }, 100);
    } catch (error) {
      console.error("Demo login error:", error);
    }
  };

  return (
    <div className="auth-container">
      <div className="auth-card">
        <h1>StoryFeed</h1>
        <h2>{isLogin ? "Sign In" : "Sign Up"}</h2>

        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <input
              type="text"
              name="username"
              placeholder="Username"
              value={formData.username}
              onChange={handleChange}
              required
            />
          </div>

          {!isLogin && (
            <div className="form-group">
              <input
                type="email"
                name="email"
                placeholder="Email"
                value={formData.email}
                onChange={handleChange}
                required
              />
            </div>
          )}

          <div className="form-group">
            <input
              type="password"
              name="password"
              placeholder="Password"
              value={formData.password}
              onChange={handleChange}
              required
            />
          </div>

          {error && <div className="error-message">{error}</div>}

          <button type="submit" disabled={loading}>
            {loading ? "Loading..." : isLogin ? "Sign In" : "Sign Up"}
          </button>
        </form>

        <div className="auth-switch">
          <p>
            {isLogin ? "Don't have an account? " : "Already have an account? "}
            <button
              type="button"
              className="link-button"
              onClick={() => setIsLogin(!isLogin)}
            >
              {isLogin ? "Sign Up" : "Sign In"}
            </button>
          </p>
        </div>

        <div className="demo-section">
          <button type="button" className="demo-button" onClick={demoLogin}>
            Use Demo Account
          </button>
        </div>
      </div>
    </div>
  );
}

export default AuthComponent;
