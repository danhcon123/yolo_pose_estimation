import subprocess
from pathlib import Path

# -------- CONFIG --------
VIDEO_NAME = "input_video.mp4"
MODEL_NAME= "yolo26n-obb.pt"
DOCKER_IMAGE = "ultralytics/ultralytics:latest"
OUTPUT_DIR = "output"


def main():
    current_dir = Path(__file__).resolve().parent
    video_path = current_dir / VIDEO_NAME

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Run Yolo26-obb with docker (with NVIDIA RTX 4700 Ti)
    docker_cmd = [
        "docker", "run", "--rm",
        "--gpus", "all",
        "--ipc=host",
        "-v", f"{current_dir}:/workspace",
        "-w", "/workspace",
        DOCKER_IMAGE,
        "yolo",
        "predict",
        f"model={MODEL_NAME}"
        f"source={VIDEO_NAME}",
        f"project={OUTPUT_DIR}",
        "name=result",
        "exist_ok=True"
    ]

    print("Running Docker command:")
    print(" ".join(docker_cmd))

    subprocess.run(docker_cmd, check=True)

    print(f"Done. Results saved in: {current_dir / OUTPUT_DIR / 'result'}")


if __name__ == "__main__":
    main()