from ultralytics import YOLO
import cv2
import numpy as np

# Load the trained model
model = YOLO("best.pt")
class_names = model.names
cap = cv2.VideoCapture('p.mp4')
# cap = cv2.VideoCapture('test.mp4')

# Resize dimensions for processing
frame_width = 1020
frame_height = 500
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))

# Distance parameters (in meters)
D_MIN = 1    # Closest distance (when object is at bottom of the frame)
D_MAX = 3    # Farthest distance (when object is at the top of the frame)

# Define colors for each class
class_colors = {
    "Pothole": (255, 0, 0),  # Blue
    "Speed Bump": (0, 255, 0)      # Green
}

while True:
    ret, img = cap.read()
    if not ret:
        break

    img = cv2.resize(img, (frame_width, frame_height))  # Ensure matching size for VideoWriter
    h, w, _ = img.shape
    results = model.predict(img)

    for r in results:
        boxes = r.boxes  # Bounding box outputs
        
    if boxes is not None:
        for box in boxes:
            # Extract bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls)
            class_name = class_names[class_id]

            # Use y2 (the bottom of the bounding box) to estimate the distance
            y_pos = y2

            # Estimate distance based on y-position (closer to bottom is closer to the camera)
            distance = D_MIN + (D_MAX - D_MIN) * (1 - y_pos / h)

            # Get color for the class
            color = class_colors.get(class_name, (255, 255, 255))  # Default to white if class not in dictionary

            # Display the distance and class name on the image
            label = f"{class_name} {distance:.2f}m"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    # Write the frame into the output video file
    out.write(img)

    # Display the frame (optional)
    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything once done
cap.release()
out.release()  # Release the VideoWriter object
cv2.destroyAllWindows()
