import cv2
import os

# Create a folder to store face images
os.makedirs("dataset", exist_ok=True)

# Initialize the webcam and face detector
camera = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

user_id = input("Enter User ID: ")  # Unique ID for the person
count = 0

print("Capturing images. Press ESC to quit.")

while True:
    ret, frame = camera.read()
    if not ret:
        print("Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        face = gray[y:y+h, x:x+w]
        face_path = f"dataset/user_{user_id}_{count}.jpg"
        cv2.imwrite(face_path, face)
        count += 1

    cv2.imshow("Capturing Faces", frame)

    key = cv2.waitKey(1)
    if key % 256 == 27:  # ESC key
        print("Exiting...")
        break

camera.release()
cv2.destroyAllWindows()
print(f"Captured {count} images for User ID {user_id}.")
