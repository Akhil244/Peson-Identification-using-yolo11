import cv2
import torch
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load YOLOv5 model
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load your trained CNN model
cnn_model = load_model('face_classifier.h5')

# Class labels (order must match training order)
class_labels = ['akhil', 'devansh', 'yashwanth']

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR frame to RGB for YOLO
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run YOLO object detection
    results = yolo_model(rgb_frame)

    # Get bounding boxes
    detections = results.xyxy[0].cpu().numpy()

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        label = int(cls)

        # Only detect 'person' class (label 0 in COCO dataset)
        if label == 0 and conf > 0.5:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            # Crop the person region safely
            face = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]

            if face.size == 0:
                continue  # skip if empty crop

            try:
                face_resized = cv2.resize(face, (128, 128))  # Resize to model input size
                face_array = img_to_array(face_resized)
                face_array = np.expand_dims(face_array, axis=0)
                face_array = face_array / 255.0  # Normalize to [0, 1]

                # Predict with CNN
                prediction = cnn_model.predict(face_array, verbose=0)
                pred_index = np.argmax(prediction)
                pred_name = class_labels[pred_index]

                # Draw bounding box and name
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, pred_name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            except Exception as e:
                print(f"Error processing face: {e}")

    # Show the frame with bounding boxes and predictions
    cv2.imshow('YOLOv5 + CNN Live Detection', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
