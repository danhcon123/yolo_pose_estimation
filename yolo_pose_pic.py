from ultralytics import YOLO
import cv2, torch


# model = YOLO("yolo26l-pose.pt")
model = YOLO("yolo26n.pt")
device = 0 if torch.cuda.is_available() else "cpu"

print("Running on:", "GPU" if device == 0 else "CPU")
source="Beispielbilder\jumping_woman.jpg" # Image input path

result = model.predict(source, device=device, save=True)
result[0].show()  # Display the image with detections
print(f"Result: {result}")
