from ultralytics import YOLO
import cv2
import numpy as np

# Load a model
model = YOLO("best.pt")
class_names = model.names
cap = cv2.VideoCapture('p.mp4')

# Distance parameters (in meters)
D_MIN = 1    # Closest distance (when pothole is at bottom of the frame)
D_MAX = 3    # Farthest distance (when pothole is at the top of the frame)

while True:
    ret, img = cap.read()
    if not ret:
        break

    img = cv2.resize(img, (1020, 500))
    h, w, _ = img.shape
    results = model.predict(img)

    for r in results:
        boxes = r.boxes  # Bounding box outputs
        
    if boxes is not None:
        for box in boxes:
            # Extract bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Use y2 (the bottom of the bounding box) to estimate the distance
            y_pos = y2

            # Estimate distance based on y-position (closer to bottom is closer to the camera)
            distance = D_MIN + (D_MAX - D_MIN) * (1 - y_pos / h)

            # Display the distance on the image
            c = class_names[int(box.cls)]
            label = f"{c} {distance:.2f}m"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                 
    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
