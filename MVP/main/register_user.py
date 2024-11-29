import sqlite3

# Connect to the database
conn = sqlite3.connect("attendance_system.db")
cursor = conn.cursor()

# Input user details
name = input("Enter the name of the user: ")

# Insert user into the Users table
cursor.execute("INSERT INTO Users (name) VALUES (?)", (name,))
conn.commit()

print(f"User '{name}' registered successfully with User ID {cursor.lastrowid}.")

# Close the connection
conn.close()
