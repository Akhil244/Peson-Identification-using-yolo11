import cv2
import torch
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load YOLOv5 model (detects persons, class 0)
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
yolo_model.conf = 0.5  # Confidence threshold for detections
yolo_model.iou = 0.45  # IOU threshold for non-max suppression

# Load CNN face classification model
cnn_model = load_model('models/mobilenet_face_classifier.keras')
class_labels = ['akhil', 'devansh', 'yashwanth']
print("Models loaded. Classes:", class_labels)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Convert frame to RGB (YOLO expects RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Perform object detection with YOLO
    results = yolo_model(rgb_frame)
    detections = results.xyxy[0].cpu().numpy()  # Get bounding box detections

    print("YOLO Detections:", detections)  # Print raw YOLO predictions for debugging

    for det in detections:
        x1, y1, x2, y2, conf, cls = det

        if int(cls) == 0 and conf > 0.5:  # Class 0 = person, confidence > 0.5
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            face = frame[y1:y2, x1:x2]  # Crop the face area

            if face.size == 0:
                print("Empty face region. Skipping.")
                continue

            # Debugging: Print face shape and the region being cropped
            print(f"Face shape: {face.shape}, x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}")

            try:
                # Resize the face region to match model input size (224x224)
                face_resized = cv2.resize(face, (224, 224))

                # Convert the face image to array and normalize it
                face_array = img_to_array(face_resized)
                face_array = np.expand_dims(face_array, axis=0) / 255.0

                # Predict the identity
                prediction = cnn_model.predict(face_array, verbose=0)

                # Debugging: Print the prediction values
                print("Prediction:", prediction)

                confidence = np.max(prediction)
                class_index = np.argmax(prediction)

                # Assign label based on confidence threshold
                if confidence < 0.7:
                    label_text = "Unknown Person"
                else:
                    label_text = f"{class_labels[class_index]} ({confidence*100:.1f}%)"

                # Debugging: Show the label and confidence for each frame
                print(f"Prediction: {label_text}")

                # Draw rectangle around detected face and label it
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36, 255, 12), 2)

            except Exception as e:
                print("Error during classification:", e)

    # Display the frame with bounding boxes and labels
    cv2.imshow('YOLOv5 + CNN Live Detection', frame)

    # Exit loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()