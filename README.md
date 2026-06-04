# YOLO Pose Estimation

Human pose estimation using [YOLOv26](https://docs.ultralytics.com/) on video input. Supports real-time display and structured keypoint output, with optional Docker-based inference.

## Scripts

| File                    | Description                                                                   |
| ----------------------- | ----------------------------------------------------------------------------- |
| `yolo_pose.py`        | Run pose estimation with live OpenCV window preview                           |
| `yolo_pose_output.py` | Run pose estimation and print detailed per-frame keypoint/box data to console |
| `obb.py`              | Oriented bounding box inference                                               |

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

### Live Preview

```bash
python yolo_pose.py
```

Runs inference on `dance_2.mp4` and displays the result in a window.

### Console Output

```bash
python yolo_pose_output.py
```

Prints bounding boxes, keypoints (xy, normalized, confidence), and a summary for each frame.

### Docker (GPU / CPU)

See [COMMAND.md](COMMAND.md) for full Docker instructions using the Ultralytics image.

**GPU (NVIDIA):**

```bash
docker run --rm --gpus all -v "${PWD}:/workspace" -w /workspace \
  ultralytics/ultralytics:latest \
  yolo pose predict model=yolo26n-pose.pt source=1.mp4 \
  project=/workspace/output name=result exist_ok=True save=True
```

**CPU:**

```bash
docker run --rm -v "${PWD}:/workspace" -w /workspace \
  ultralytics/ultralytics:latest-cpu \
  yolo pose predict model=yolo26n-pose.pt source=1.mp4 \
  project=/workspace/output name=result exist_ok=True save=True device=cpu
```

## Project Structure

```
yolo_pose_estimation/
├── yolo_pose.py           # Live preview script
├── yolo_pose_output.py    # Console output script
├── obb.py                 # OBB inference script
├── requirements.txt
├── COMMAND.md             # Docker usage guide
├── input/                 # Input videos (not tracked)
├── output/result/         # Saved prediction videos (not tracked)
├── docu/                  # Assignment and presentation files
└── publication/           # Reference papers
```
