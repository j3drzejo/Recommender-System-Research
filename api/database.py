import sqlite3
from datetime import datetime

DB_PATH = './db.db'

def init_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            userId INTEGER NOT NULL,
            videoId INTEGER NOT NULL,
            watched_percent INTEGER CHECK(watched_percent >= 0 AND watched_percent <= 100),
            liked INTEGER CHECK(liked IN (-1, 0, 1)),
            whenReacted INTEGER CHECK(whenReacted >= 0 AND whenReacted <= 100),
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
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