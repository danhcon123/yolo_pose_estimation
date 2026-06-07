"""
Inference with Pose estimation YOLO26
Display the inferenced frames
"""
from ultralytics import YOLO
import cv2, torch


model = YOLO("yolo26l-pose.pt")
# model = YOLO("yolo26n-pose_openvino_model")
device = 0 if torch.cuda.is_available() else "cpu"

print("Running on:", "GPU" if device == 0 else "CPU")
source="input/dance_2.mp4" # Video input path

result = model.predict(source, stream=True, device=device)
print(f"Result: {result}")

for r in result:
    frame = r.plot()
    frame = cv2.resize(frame, (960, 540))  # change display size here

    # Show video
    cv2.imshow("YOLO26 Pose", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()