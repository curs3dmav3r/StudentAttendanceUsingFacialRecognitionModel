import cv2
import sqlite3
import datetime

# Connect to the database
conn = sqlite3.connect("attendance_system.db")
cursor = conn.cursor()

# Log attendance in the database
def log_attendance(user_id):
    cursor.execute("INSERT INTO Attendance (user_id) VALUES (?)", (user_id,))
    conn.commit()
    print(f"Attendance logged for User ID {user_id} at {datetime.datetime.now()}.")

# Close the connection when done
def close_connection():
    conn.close()

# Start webcam
camera = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trained_model/face_trainer.yml")

marked_users = set()  # Track users already logged

print("Press ESC to exit.")

while True:
    ret, frame = camera.read()
    if not ret:
        print("Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        label, confidence = recognizer.predict(face)

        if confidence < 50:
            user_id = label  # User ID corresponds to the label
            if user_id not in marked_users:
                log_attendance(user_id)  # Log attendance in the database
                marked_users.add(user_id)
        else:
            user_id = "Unknown"

        # Draw rectangle and label on the frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, str(user_id), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Face Recognition with Attendance", frame)

    key = cv2.waitKey(1)
    if key % 256 == 27:  # ESC key
        print("Exiting...")
        break

camera.release()
cv2.destroyAllWindows()
close_connection()
