# import torch
# import numpy as np
import cv2
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cosine

# Load InsightFace model
def load_insightface_model():
    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app

# Detect faces using InsightFace
def detect_faces(insightface_model, image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = insightface_model.get(image_rgb)
    
    face_data = []
    for face in faces:
        x1, y1, x2, y2 = map(int, face.bbox)
        face_crop = image[y1:y2, x1:x2]
        face_data.append((face_crop, (x1, y1, x2, y2), face.embedding))

    return face_data

# Find the most similar face using cosine similarity
def find_best_match(input_feature, detected_faces):
    best_match = None
    best_similarity = -1

    for face_crop, bbox, feature in detected_faces:
        similarity = 1 - cosine(input_feature, feature)
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = (bbox, similarity)

    return best_match

# Calculate accuracy
def calculate_accuracy(total_faces_detected, correct_matches):
    return (correct_matches / total_faces_detected) * 100 if total_faces_detected > 0 else 0.0

if __name__ == "__main__":
    # Load InsightFace model
    insightface_model = load_insightface_model()

    # Load and extract features from input face (Reference Image)
    input_face = cv2.imread(r"path to your input image")  # Update with correct path
    input_feature = detect_faces(insightface_model, input_face)[0][-1]  # Extract embedding

    # **Use Video Instead of Webcam**
    video_path = r"path to your input video to find the person"  # Update with the correct video file path
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Failed to open video file!")
        exit()

    total_faces_detected = 0
    correct_matches = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Stop when video ends

        detected_faces = detect_faces(insightface_model, frame)

        if detected_faces:
            total_faces_detected += len(detected_faces)

            best_match = find_best_match(input_feature, detected_faces)

            if best_match:
                (x1, y1, x2, y2), similarity = best_match

                if similarity > 0.8:  # Adjust threshold if needed
                    correct_matches += 1

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 6)
                cv2.putText(frame, f"Match: {similarity:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)
                
            display_width = 800  # Set a fixed width
            aspect_ratio = display_width / frame.shape[1]
            display_height = int(frame.shape[0] * aspect_ratio)
            resized_frame = cv2.resize(frame, (display_width, display_height))
        
        # Display the video frame
        cv2.imshow("Video Face Matching", resized_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    accuracy = calculate_accuracy(total_faces_detected, correct_matches)
    print(f"\nAccuracy: {accuracy:.2f}%")

    cap.release()
    cv2.destroyAllWindows()
