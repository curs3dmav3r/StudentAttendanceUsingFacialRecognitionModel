import os
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Load YOLOv8 model
yolo_model = YOLO('C:/Users/kritartha dhakal/Desktop/yolov8n-face-lindevs.pt')  # Use a pre-trained YOLOv8 model trained for face detection

# Load pre-trained FaceNet model
model_path = "C:/Users/kritartha dhakal/Downloads/facenet-tensorflow-tensorflow2-default-v2"
facenet_model = tf.saved_model.load(model_path)
infer = facenet_model.signatures['serving_default']

# Function to detect and crop faces using YOLO
def detect_and_crop_face(image):
    results = yolo_model(image)
    boxes = results[0].boxes.xyxy.numpy()  # Bounding boxes [x1, y1, x2, y2]
    if len(boxes) == 0:
        raise ValueError("No face detected in the captured frame")
    
    # Use the first detected face (assuming one face per frame)
    x1, y1, x2, y2 = map(int, boxes[0])
    cropped_face = image[y1:y2, x1:x2]
    cropped_face = cv2.resize(cropped_face, (160, 160))  # FaceNet input size
    return cropped_face

# Function to preprocess the captured frame for the model
def preprocess_image(image):
    face = detect_and_crop_face(image)
    face = face.astype('float32') / 255.0  # Normalize pixel values
    face = np.expand_dims(face, axis=0)  # Add batch dimension
    return face

# Function to extract features using FaceNet
# def extract_features(image):
#     img = preprocess_image(image)
#     output = infer(input_1=tf.convert_to_tensor(img))
#     embedding = output['Bottleneck_BatchNorm']  # Extract features
#     return embedding / np.linalg.norm(embedding)  # Normalize the vector

def extract_features(image):
    img = preprocess_image(image)
    output = infer(input_1=tf.convert_to_tensor(img))
    embedding = output['Bottleneck_BatchNorm']  # Extract features
    embedding = tf.squeeze(embedding)  # Remove extra dimensions if any
    return embedding.numpy() / np.linalg.norm(embedding.numpy()) 

# Function to find the most similar image
def find_most_similar(captured_features, dataset_features, dataset_paths):
    similarities = cosine_similarity([captured_features], dataset_features)[0]
    best_match_idx = np.argmax(similarities)
    return dataset_paths[best_match_idx], similarities[best_match_idx]

# Real-time image capture function
def capture_image():
    cap = cv2.VideoCapture(0)  # Open webcam (0 for default camera)
    print("Press 'Spacebar' to capture an image, or 'Esc' to exit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image. Please check your camera.")
            break
        
        # Show the video feed
        cv2.imshow("Live Feed - Press 'Spacebar' to Capture", frame)
        
        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # Spacebar to capture
            captured_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cap.release()
            cv2.destroyAllWindows()
            return captured_image
        elif key == 27:  # Esc to exit
            cap.release()
            cv2.destroyAllWindows()
            exit("Exiting the program.")
            
# Dataset preparation
dataset_folder = "C:/Users/kritartha dhakal/Desktop/tryImages"
dataset_features = []
dataset_paths = []

for filename in os.listdir(dataset_folder):
    image_path = os.path.join(dataset_folder, filename)
    if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        features = extract_features(img)
        dataset_features.append(features)
        dataset_paths.append(image_path)

dataset_features = np.array(dataset_features)
print("Feature extraction for dataset complete.")

# Real-time comparison
print("Starting live capture...")
captured_image = capture_image()  # Capture an image from the webcam
try:
    captured_features = extract_features(captured_image)
    predicted_image_path, similarity_score = find_most_similar(captured_features, dataset_features, dataset_paths)
    
    # Display results
    print(f"Predicted Similar Image: {predicted_image_path} (Similarity: {similarity_score:.4f})")
    predicted_image = cv2.cvtColor(cv2.imread(predicted_image_path), cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(captured_image)
    plt.title("Captured Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(predicted_image)
    plt.title("Predicted Image")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

except ValueError as e:
    print(str(e))
