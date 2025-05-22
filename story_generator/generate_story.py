import ollama
import json
import sqlite3
import time
import os

# Ensure the directory for the database exists
db_path = './db.db'
os.makedirs(os.path.dirname(db_path), exist_ok=True)

# Connect to the database (will create if it doesn't exist)
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create tables if they don't exist
cursor.execute('''
CREATE TABLE IF NOT EXISTS videos (
    videoId INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL
);
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS labels (
    videoId INTEGER,
    label TEXT NOT NULL,
    FOREIGN KEY (videoId) REFERENCES videos(videoId)
);
''')

# Prompt for story generation
prompt = (
    "Write a short, highly engaging story (200–400 words) in the style of a Reddit post, told in first person by someone "
    "who personally experienced the events. The narrator's voice must be authentic and casual. The story should be plausible, but can be exaggerated for humor or drama. "
    "After the story, return a list of 10–15 descriptive labels relevant to the story. "
    "Output only a valid JSON object with exactly two keys: \"story\" and \"labels\". "
    "\"story\" must be a string. \"labels\" must be a list of 10–15 strings. "
    "Do not include any extra text, formatting, or markdown. Only output the raw JSON object."
)

# Generate and store stories
for i in range(100):  # Change to 100 if needed
    try:
        response = ollama.chat(
            model='gemma3:4b',
            messages=[{'role': 'user', 'content': prompt}]
        )

        content = response['message']['content'].strip()
        content = content.replace('```json', '').replace('```', '')  # Strip formatting artifacts

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as json_err:
            print(f"[{i+1}/100] JSON parsing failed: {json_err}")
            continue

        story = parsed['story']
        labels = parsed['labels']

        # Insert story
        cursor.execute('INSERT INTO videos (text) VALUES (?)', (story,))
        video_id = cursor.lastrowid

        # Insert labels
        for label in labels:
            cursor.execute('INSERT INTO labels (videoId, label) VALUES (?, ?)', (video_id, label))

        print(f"[{i+1}/100] Inserted story with videoId {video_id}")
        time.sleep(0.5)

    except Exception as e:
        print(f"[{i+1}/100] Unexpected error: {e}")

# Finalize
conn.commit()
conn.close()
