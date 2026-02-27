Live Object Detection using OpenCV and YOLOv8

--------------------------------------------------
Project Overview
--------------------------------------------------

This project performs real-time object detection using:

- OpenCV (cv2) for video capture and display
- YOLOv8 (Ultralytics) for object detection
- Pretrained COCO dataset model

The system captures live video from a webcam and detects
objects in real time with bounding boxes and class labels.


--------------------------------------------------
Model Used
--------------------------------------------------

YOLOv8 Nano (yolov8n.pt)

- Lightweight
- Fast inference
- Suitable for CPU and edge devices
- Trained on COCO dataset (80 classes)


--------------------------------------------------
Requirements
--------------------------------------------------

Install dependencies:

pip install ultralytics opencv-python

Verify installation:

python -c "import cv2; from ultralytics import YOLO"


--------------------------------------------------
Project Structure
--------------------------------------------------

live-object-detection/

    detect.py
    README.md


--------------------------------------------------
Python Code (detect.py)
--------------------------------------------------

import cv2
from ultralytics import YOLO

# Load YOLOv8 nano model
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run object detection
    results = model(frame)

    # Draw bounding boxes and labels
    annotated_frame = results[0].plot()

    # Display output
    cv2.imshow("Live Object Detection", annotated_frame)

    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


--------------------------------------------------
How to Run
--------------------------------------------------

1. Save the code as detect.py
2. Open terminal inside project folder
3. Run:

   python detect.py

Press 'q' to exit the application.


--------------------------------------------------
Supported Object Classes
--------------------------------------------------

The model is trained on the COCO dataset, which includes:

Person
Car
Bicycle
Dog
Cat
Bus
Truck
Bottle
Chair
Laptop
... and more (80 classes total)


--------------------------------------------------
Performance Notes
--------------------------------------------------

Model Comparison:

yolov8n  - Very Fast  - Best for CPU / Raspberry Pi
yolov8s  - Fast       - Good for Laptop
yolov8m  - Medium     - Good for GPU
yolov8l  - Slower     - High-end GPU

To change model:

model = YOLO("yolov8s.pt")


--------------------------------------------------
Customization
--------------------------------------------------

Detect only person:

results = model(frame, classes=[0])

Resize frame for faster processing:

frame = cv2.resize(frame, (640, 480))

Enable object tracking:

results = model.track(frame, persist=True)


--------------------------------------------------
Possible Extensions
--------------------------------------------------

- Object counting
- Line crossing detection
- Face recognition
- Custom trained YOLO model
- Deployment on Jetson / Raspberry Pi
- Integration with Triton Inference Server


--------------------------------------------------
Tested Environment
--------------------------------------------------

Python 3.8+
OpenCV 4.x
Ultralytics YOLOv8
Windows / Linux

