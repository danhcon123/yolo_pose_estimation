"""
Inference with Pose Estimation YOLO26
Display input + output frames side by side
Save the combined video
"""

from ultralytics import YOLO
import cv2
import torch
from pathlib import Path


# -----------------------------
# Config
# -----------------------------
model_path = "yolo26n-pose.pt"
source = "input/dance_2.mp4"
output_path = "output/dance_1_yolo26_pose_comparison.mp4"

display_width = 960
display_height = 540

Path("output").mkdir(exist_ok=True)


# -----------------------------
# Load model
# -----------------------------
model = YOLO(model_path)

device = 0 if torch.cuda.is_available() else "cpu"
print("Running on:", "GPU" if device == 0 else "CPU")


# -----------------------------
# Get video FPS
# -----------------------------
cap = cv2.VideoCapture(source)
fps = cap.get(cv2.CAP_PROP_FPS)

if fps == 0:
    fps = 30  # fallback if FPS cannot be read

cap.release()


# Combined frame: input + output
combined_width = display_width * 2
combined_height = display_height


# -----------------------------
# Video writer
# -----------------------------
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (combined_width, combined_height)
)


# -----------------------------
# Inference
# -----------------------------
results = model.predict(
    source,
    stream=True,
    device=device
)

for r in results:
    # Original input frame
    input_frame = r.orig_img.copy()

    # YOLO pose output frame
    output_frame = r.plot()

    # Resize both frames to same display size
    input_frame = cv2.resize(input_frame, (display_width, display_height))
    output_frame = cv2.resize(output_frame, (display_width, display_height))

    # Add labels
    cv2.putText(
        input_frame,
        "Input",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.3,
        (255, 255, 255),
        3
    )

    cv2.putText(
        output_frame,
        "YOLO26 Pose Output",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.3,
        (255, 255, 255),
        3
    )

    # Combine side by side
    combined_frame = cv2.hconcat([input_frame, output_frame])

    # Save frame to output video
    writer.write(combined_frame)

    # Show video
    cv2.imshow("Input vs YOLO26 Pose", combined_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# -----------------------------
# Cleanup
# -----------------------------
writer.release()
cv2.destroyAllWindows()

print(f"Saved video to: {output_path}")