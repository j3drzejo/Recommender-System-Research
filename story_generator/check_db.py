import sqlite3

# Connect to the database
conn = sqlite3.connect('../db.db')
cursor = conn.cursor()

# Execute the query to fetch all videos and labels
query = '''
SELECT *
FROM videos
'''

cursor.execute(query)

# Fetch all results and display them
results = cursor.fetchall()

print(results)
# Close the connection
conn.close()
