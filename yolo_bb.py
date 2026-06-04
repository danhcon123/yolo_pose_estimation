"""
Inference with Pose estimation YOLO26
Display the inferenced frames
"""
from ultralytics import YOLO


model = YOLO("yolo26n.pt")

# result = model.predict(source="image_1.png", save=True)
model.show(model.predict(source="image_1.png", save=True))