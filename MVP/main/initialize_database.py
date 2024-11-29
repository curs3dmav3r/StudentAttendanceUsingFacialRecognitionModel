import sqlite3

# Connect to SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect("attendance_system.db")
cursor = conn.cursor()

# Create Users table
cursor.execute('''
CREATE TABLE IF NOT EXISTS Users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL
)
''')

# Create Attendance table
cursor.execute('''
CREATE TABLE IF NOT EXISTS Attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES Users (user_id)
)
''')

print("Database initialized successfully.")

# Commit changes and close the connection
conn.commit()
conn.close()
