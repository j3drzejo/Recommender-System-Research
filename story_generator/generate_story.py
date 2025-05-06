import ollama
import ast
import sqlite3

conn = sqlite3.connect('../db.db')
cursor = conn.cursor()

# Create the 'videos' table
create_videos_table_query = '''
CREATE TABLE IF NOT EXISTS videos (
    videoId INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL
);
'''

# Create the 'labels' table
create_labels_table_query = '''
CREATE TABLE IF NOT EXISTS labels (
    videoId INTEGER,
    label TEXT NOT NULL,
    FOREIGN KEY (videoId) REFERENCES videos(videoId)
);
'''

# Generate response from the AI model
response = ollama.chat(
    model='gemma3:4b',
    messages=[
        {'role': 'user', 'content': "Write a short reddit-like original story (200–400 words) on a random topic, and generate 10–15 descriptive labels for the story. Output only a valid Python dictionary with exactly two keys: 'story' and 'labels'. 'story': The story string (200–400 words). 'labels': A list of 10–15 descriptive labels (each a single word or short phrase). Use single quotes for all strings (keys and values). Do NOT output any additional text, explanation, or formatting (no JSON, no markdown, no code fences). Ensure the output is valid Python syntax (parsable with eval() or ast.literal_eval())."}
    ]
)

# Clean up the response string (remove unwanted markdown/code block)
cleaned_response = response['message']['content'].strip().replace('```python', '').replace('```', '')

# Parse the cleaned string to get the story and labels
parsed_response = ast.literal_eval(cleaned_response)

story_text = parsed_response['story']
labels = parsed_response['labels']

# Step 1: Execute table creation queries
cursor.execute(create_videos_table_query)
cursor.execute(create_labels_table_query)

# Step 2: Insert the video and labels data into the database

# Insert the video story text into the 'videos' table
cursor.execute('INSERT INTO videos (text) VALUES (?)', (story_text,))
video_id = cursor.lastrowid  # Get the last inserted videoId

# Insert each label into the 'labels' table with the corresponding videoId
for label in labels:
    cursor.execute('INSERT INTO labels (videoId, label) VALUES (?, ?)', (video_id, label))

# Step 3: Commit the changes and close the connection
conn.commit()
conn.close()

# Optionally print the inserted data to verify
print(f"Video with ID {video_id} inserted with story: {story_text}")
for label in labels:
    print(f"Label for videoId {video_id}: {label}")
