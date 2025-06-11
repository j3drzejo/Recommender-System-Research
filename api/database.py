import sqlite3
import hashlib
import secrets
import json
from datetime import datetime

DB_PATH = './db.db'

def init_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Users table for authentication
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_login DATETIME,
            is_active BOOLEAN DEFAULT 1
        )
    ''')

    # Model performance tracking
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_type TEXT NOT NULL,
            user_id INTEGER,
            video_id INTEGER,
            predicted_score REAL,
            actual_interaction INTEGER,
            accuracy_score REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')

    # Bandit arms persistence
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS bandit_arms (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER NOT NULL,
            total_count INTEGER DEFAULT 0,
            total_reward REAL DEFAULT 0.0,
            avg_reward REAL DEFAULT 0.0,
            last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(video_id)
        )
    ''')

    # User preferences and embeddings
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            embedding_vector TEXT,  -- JSON string of the embedding
            model_version TEXT DEFAULT 'v1',
            last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id),
            UNIQUE(user_id, model_version)
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            userId INTEGER NOT NULL,
            videoId INTEGER NOT NULL,
            watched_percent INTEGER CHECK(watched_percent >= 0 AND watched_percent <= 100),
            liked INTEGER CHECK(liked IN (-1, 0, 1)),
            whenReacted INTEGER CHECK(whenReacted >= 0 AND whenReacted <= 100),
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (userId) REFERENCES users (id),
            UNIQUE(userId, videoId)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS labels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            videoId INTEGER NOT NULL,
            label TEXT NOT NULL,
            FOREIGN KEY (videoId) REFERENCES videos (videoId)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS videos (
            videoId INTEGER PRIMARY KEY,
            text TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

def save_interaction(interaction_data):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        INSERT OR REPLACE INTO interactions 
        (userId, videoId, watched_percent, liked, whenReacted, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        interaction_data.userId,
        interaction_data.videoId,
        interaction_data.watched_percent,
        interaction_data.liked,
        interaction_data.whenReacted,
        datetime.now()
    ))
    
    conn.commit()
    conn.close()

def get_stats():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('SELECT COUNT(*) FROM interactions')
    interaction_count = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM videos')
    video_count = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(DISTINCT userId) FROM interactions')
    user_count = cursor.fetchone()[0]
    
    conn.close()
    
    return {
        "total_interactions": interaction_count,
        "total_videos": video_count,
        "total_users": user_count
    }

def create_user(username, email, password):
    """Create a new user account"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Hash password
        salt = secrets.token_hex(16)
        password_hash = hashlib.sha256((password + salt).encode()).hexdigest() + ":" + salt
        
        cursor.execute('''
            INSERT INTO users (username, email, password_hash)
            VALUES (?, ?, ?)
        ''', (username, email, password_hash))
        
        user_id = cursor.lastrowid
        conn.commit()
        return {"success": True, "user_id": user_id, "message": "User created successfully"}
    
    except sqlite3.IntegrityError as e:
        return {"success": False, "message": "Username or email already exists"}
    except Exception as e:
        return {"success": False, "message": str(e)}
    finally:
        conn.close()

def authenticate_user(username, password):
    """Authenticate user login"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            SELECT id, username, email, password_hash, is_active
            FROM users 
            WHERE username = ? AND is_active = 1
        ''', (username,))
        
        user = cursor.fetchone()
        if not user:
            return {"success": False, "message": "Invalid username or password"}
        
        user_id, username, email, stored_hash, is_active = user
        
        # Verify password
        if ":" in stored_hash:
            hash_part, salt = stored_hash.split(":", 1)
            password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
            
            if password_hash == hash_part:
                # Update last login
                cursor.execute('''
                    UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?
                ''', (user_id,))
                conn.commit()
                
                return {
                    "success": True, 
                    "user": {
                        "id": user_id,
                        "username": username,
                        "email": email
                    }
                }
        
        return {"success": False, "message": "Invalid username or password"}
    
    except Exception as e:
        return {"success": False, "message": str(e)}
    finally:
        conn.close()

def save_model_performance(model_type, user_id, video_id, predicted_score, actual_interaction):
    """Save model performance metrics for analysis"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Calculate accuracy based on interaction
        accuracy = 0.0
        if actual_interaction == 1:  # Liked
            accuracy = max(0, predicted_score)  # Higher score = better prediction
        elif actual_interaction == -1:  # Disliked
            accuracy = max(0, 1.0 - predicted_score)  # Lower score = better prediction
        else:  # Neutral
            accuracy = 1.0 - abs(predicted_score - 0.5)  # Mid-range score = better
        
        cursor.execute('''
            INSERT INTO model_performance 
            (model_type, user_id, video_id, predicted_score, actual_interaction, accuracy_score)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (model_type, user_id, video_id, predicted_score, actual_interaction, accuracy))
        
        conn.commit()
    except Exception as e:
        print(f"Error saving model performance: {e}")
    finally:
        conn.close()

def update_bandit_arm(video_id, reward):
    """Update bandit arm statistics in database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Get current stats
        cursor.execute('SELECT total_count, total_reward FROM bandit_arms WHERE video_id = ?', (video_id,))
        result = cursor.fetchone()
        
        if result:
            count, total_reward = result
            new_count = count + 1
            new_total = total_reward + reward
            new_avg = new_total / new_count
            
            cursor.execute('''
                UPDATE bandit_arms 
                SET total_count = ?, total_reward = ?, avg_reward = ?, last_updated = CURRENT_TIMESTAMP
                WHERE video_id = ?
            ''', (new_count, new_total, new_avg, video_id))
        else:
            cursor.execute('''
                INSERT INTO bandit_arms (video_id, total_count, total_reward, avg_reward)
                VALUES (?, 1, ?, ?)
            ''', (video_id, reward, reward))
        
        conn.commit()
    except Exception as e:
        print(f"Error updating bandit arm: {e}")
    finally:
        conn.close()

def get_bandit_arms():
    """Get all bandit arm statistics"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute('SELECT video_id, total_count, avg_reward FROM bandit_arms')
        return {video_id: {'count': count, 'avg_reward': avg_reward} 
                for video_id, count, avg_reward in cursor.fetchall()}
    except Exception as e:
        print(f"Error getting bandit arms: {e}")
        return {}
    finally:
        conn.close()

def save_user_embedding(user_id, embedding_vector, model_version='v1'):
    """Save user embedding to database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        embedding_json = json.dumps(embedding_vector.tolist() if hasattr(embedding_vector, 'tolist') else embedding_vector)
        
        cursor.execute('''
            INSERT OR REPLACE INTO user_embeddings (user_id, embedding_vector, model_version)
            VALUES (?, ?, ?)
        ''', (user_id, embedding_json, model_version))
        
        conn.commit()
    except Exception as e:
        print(f"Error saving user embedding: {e}")
    finally:
        conn.close()

def get_user_embedding(user_id, model_version='v1'):
    """Get user embedding from database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            SELECT embedding_vector FROM user_embeddings 
            WHERE user_id = ? AND model_version = ?
        ''', (user_id, model_version))
        
        result = cursor.fetchone()
        if result:
            return json.loads(result[0])
        return None
    except Exception as e:
        print(f"Error getting user embedding: {e}")
        return None
    finally:
        conn.close()