import cv2
import torch
from ultralytics import YOLO

# Load the YOLOv8 model (make sure to download the model if you haven't already)
model = YOLO("../YOLO/yolov8n.pt")

# Load the 360-degree image
image_path = "../Full 360 Images/bend5.png"
image = cv2.imread(image_path)

if image is None:
    print("Error: Could not load image. Check the path.")
    exit()

# Perform object detection
results = model(image)

# Draw bounding boxes on the image
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
        confidence = box.conf[0].item()  # Confidence score
        class_id = int(box.cls[0].item())  # Class ID
        label = f"{model.names[class_id]}: {confidence:.2f}"

        # Draw the bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Show the result
cv2.imshow("YOLOv8 360 Image Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
