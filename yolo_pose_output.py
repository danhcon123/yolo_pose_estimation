from ultralytics import YOLO
import torch
import json

model = YOLO("yolo26n-pose.pt")
device = 0 if torch.cuda.is_available() else "cpu"

results = model.predict(
    source="input\dance_2.mp4",
    stream=True,
    device=device,
    verbose=False
)

print("Running on:", "GPU" if device == 0 else "CPU")

for frame_id, r in enumerate(results):
    print("\n" + "=" * 80)
    print(f"FRAME {frame_id}")
    print("=" * 80)

    print("Path:", r.path)
    print("Original image shape:", r.orig_shape)
    print("Class names:", r.names)
    print("Speed:", r.speed)

    # Bounding boxes
    if r.boxes is not None:
        print("\n--- BOXES ---")
        print("Raw box data:")
        print(r.boxes.data.cpu().numpy())

        print("xyxy:")
        print(r.boxes.xyxy.cpu().numpy())

        print("xywh:")
        print(r.boxes.xywh.cpu().numpy())

        print("Confidence:")
        print(r.boxes.conf.cpu().numpy())

        print("Class IDs:")
        print(r.boxes.cls.cpu().numpy())

    # Pose keypoints
    if r.keypoints is not None:
        print("\n--- KEYPOINTS ---")
        print("Raw keypoint data:")
        print(r.keypoints.data.cpu().numpy())

        print("Keypoints xy:")
        print(r.keypoints.xy.cpu().numpy())

        print("Keypoints normalized xyn:")
        print(r.keypoints.xyn.cpu().numpy())

        if r.keypoints.conf is not None:
            print("Keypoint visibility:")
            print(r.keypoints.conf.cpu().numpy())

    # Masks, only relevant for segmentation models
    if r.masks is not None:
        print("\n--- MASKS ---")
        print(r.masks.data.cpu().numpy())

    # Classification probabilities, only relevant for classification models
    if r.probs is not None:
        print("\n--- PROBS ---")
        print(r.probs.data.cpu().numpy())

    # Oriented bounding boxes, only relevant for OBB models
    if r.obb is not None:
        print("\n--- OBB ---")
        print(r.obb.data.cpu().numpy())

    # Compact summary
    print("\n--- SUMMARY ---")
    print(r.summary())

    # Show frame with detections
    r.show()

    # Stop early for testing
    if frame_id == 0:
        break