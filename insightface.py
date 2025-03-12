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
    input_face = cv2.imread(r"path to your input image")
    input_feature = detect_faces(insightface_model, input_face)[0][-1]

    # Start webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open webcam!")
        exit()

    total_faces_detected = 0
    correct_matches = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame!")
            break

        detected_faces = detect_faces(insightface_model, frame)

        if detected_faces:
            total_faces_detected += len(detected_faces)

            best_match = find_best_match(input_feature, detected_faces)

            if best_match:
                (x1, y1, x2, y2), similarity = best_match

                if similarity > 0.8:
                    correct_matches += 1

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 6)
                cv2.putText(frame, f"Match: {similarity:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)
        
        cv2.imshow("Webcam Face Matching", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    accuracy = calculate_accuracy(total_faces_detected, correct_matches)
    print(f"\nAccuracy: {accuracy:.2f}%")

    cap.release()
    cv2.destroyAllWindows()
