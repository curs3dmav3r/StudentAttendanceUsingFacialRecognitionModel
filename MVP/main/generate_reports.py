import pandas as pd

# Load the attendance data
attendance_file = "attendance.csv"

# Check if the file exists
try:
    df = pd.read_csv(attendance_file)
except FileNotFoundError:
    print(f"{attendance_file} not found! Please run the attendance script first.")
    exit()

# Generate daily report
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Date'] = df['Timestamp'].dt.date

# Group by User ID and Date
daily_report = df.groupby(['User ID', 'Date']).size().reset_index(name='Attendance Count')

# Save the report to a new file
report_file = "daily_attendance_report.csv"
daily_report.to_csv(report_file, index=False)

print(f"Daily attendance report saved as {report_file}.")
