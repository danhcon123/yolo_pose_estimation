"""
Inference with Pose estimation YOLO26
Display and save the inferenced frames
"""

from ultralytics import YOLO
import cv2, torch
import os

model = YOLO("yolo26l-pose.pt")
device = 0 if torch.cuda.is_available() else "cpu"

print("Running on:", "GPU" if device == 0 else "CPU")

source = "input/soccer.mp4"
output_path = "output/soccer_pose_output.mp4"
os.makedirs("output", exist_ok=True)

# Get input video FPS
cap = cv2.VideoCapture(source)
fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()

if fps == 0:
    fps = 30

# Important: size format is (width, height)
output_size = (540, 960)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(output_path, fourcc, fps, output_size)

result = model.predict(source, stream=True, device=device)

for r in result:
    frame = r.plot(
       kpt_radius=15,   # bigger keypoints
        line_width=10
    )

    frame = cv2.resize(frame, output_size)

    # Save frame to output video
    writer.write(frame)

    # Show video
    cv2.imshow("YOLO26 Pose", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

writer.release()
cv2.destroyAllWindows()

print(f"Saved video to: {output_path}")