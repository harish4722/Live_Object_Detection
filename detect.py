import cv2
from ultralytics import YOLO

# Load YOLO model (nano model - fast)
model = YOLO("yolov8n.pt")   # downloads automatically first time

# Open camera
cap = cv2.VideoCapture(0)   # 0 = default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame)

    # Draw results
    annotated_frame = results[0].plot()

    # Show output
    cv2.imshow("Live Object Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()