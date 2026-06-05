# Inference with Docker

## NVIDIA

Requirements
Window, Docker Desktop, Ultralytics Docker image, NVIDIA GPU + NVIDIA Docker/GPU support only if using GPU inference

```
docker pull ultralytics/ultralytics:latest
```

Folder structure

```
my_yolo_project/
│
├── 1.mp4
├── obb.py
└── output/
    └── result/
        └── ... saved prediction video after inference
```

Check NVIDIA GPU Inside Docker

```
docker run --rm --gpus all ultralytics/ultralytics:latest nvidia-smi
```

Predict (Pose Estimation) with Docker

```
# Open folder
cd C:\Users\YourName\Desktop\my_yolo_project

# Inference with NVIDIA                                         
docker run --rm --gpus all -v "${PWD}:/workspace" -w /workspace ultralytics/ultralytics:latest yolo pose predict model=yolo26n-pose.pt source=1.mp4 project=/workspace/output name=result exist_ok=True save=True show=False

# Open saved video folder
explorer .\output\result  
```

---

## CPU

Requirements
Windows, Docker Desktop, Ultralytics Docker CPU image

```bash
docker pull ultralytics/ultralytics:latest-cpu
```

Predict with CPU

```bash
# Open folder
cd C:\Users\YourName\Desktop\my_yolo_project

# Inference on CPU
docker run --rm -v "${PWD}:/workspace" -w /workspace ultralytics/ultralytics:latest-cpu yolo pose predict model=yolo26n-pose.pt source=1.mp4 project=/workspace/output name=result exist_ok=True save=True show=False device=cpu

# Open saved video folder
explorer .\output\result
```

Folder structure

```text
my_yolo_project/
│
├── 1.mp4
├── obb.py                 # optional, only needed if you run a custom Python script
└── output/
    └── result/
        └── ... saved prediction video after inference
```


## Command

Convert in Onnx command:

```yolo
yolo export model=yolo26n-pose.pt format=onnx imgsz=640
```
