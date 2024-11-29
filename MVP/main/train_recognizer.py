import cv2
import os
import numpy as np

# Load the dataset folder
dataset_path = "dataset"
recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

faces = []
labels = []

# Load all images and labels
for filename in os.listdir(dataset_path):
    if filename.endswith(".jpg"):
        label = int(filename.split("_")[1])  # Extract user ID from filename
        img_path = os.path.join(dataset_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        faces.append(img)
        labels.append(label)

# Train the recognizer
print("Training recognizer...")
recognizer.train(faces, np.array(labels))
recognizer.save("trained_model/face_trainer.yml")
print("Training complete. Model saved in 'trained_model/face_trainer.yml'.")
