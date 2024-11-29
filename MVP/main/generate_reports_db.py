import sqlite3
import pandas as pd

# Connect to the database
conn = sqlite3.connect("attendance_system.db")

# Query attendance data
query = '''
SELECT Users.name, DATE(Attendance.timestamp) as date, COUNT(Attendance.id) as attendance_count
FROM Attendance
JOIN Users ON Attendance.user_id = Users.user_id
GROUP BY Users.name, DATE(Attendance.timestamp)
'''

df = pd.read_sql_query(query, conn)

# Save the report as a CSV file
report_file = "daily_attendance_report_db.csv"
df.to_csv(report_file, index=False)

print(f"Daily attendance report saved as {report_file}.")

# Close the connection
conn.close()
