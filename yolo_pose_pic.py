from ultralytics import YOLO
import cv2, torch


model = YOLO("yolo26l-pose.pt")
device = 0 if torch.cuda.is_available() else "cpu"

print("Running on:", "GPU" if device == 0 else "CPU")
source="input/running_man.jpg" # Image input path

result = model.predict(source, device=device, save=True)
result[0].show()  # Display the image with detections
print(f"Result: {result}")
