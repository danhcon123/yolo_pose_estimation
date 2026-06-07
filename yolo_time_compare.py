from __future__ import annotations

import csv
import os
import platform
import statistics
import subprocess
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from ultralytics import YOLO


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"

MODELS = {
    "YOLO11n-Pose": "yolo11l-pose.pt",
    "YOLO26n-Pose": "yolo26l-pose.pt",
}

DEVICE = "cpu"
IMAGE_SIZE = 640
EXPECTED_IMAGE_COUNT = 10
WARMUP_RUNS = 3

# Set to 1 for a minimal test.
# Increase to 3 or 5 for more stable measurements.
REPEATS_PER_IMAGE = 1


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def get_cpu_name() -> str:
    """Return a readable CPU name on Windows, Linux, or macOS."""
    system = platform.system()

    try:
        if system == "Windows":
            output = subprocess.check_output(
                ["powershell", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            return output.strip()

        if system == "Linux":
            cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8")
            for line in cpuinfo.splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()

        if system == "Darwin":
            output = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            return output.strip()

    except Exception:
        pass

    return platform.processor() or "Unknown CPU"


def get_ram_gb() -> float | None:
    """Return total RAM in GB without requiring an additional package."""
    try:
        if platform.system() == "Windows":
            output = subprocess.check_output(
                [
                    "powershell",
                    "-Command",
                    "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            return int(output.strip()) / (1024**3)

        if hasattr(os, "sysconf"):
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return (pages * page_size) / (1024**3)

    except Exception:
        pass

    return None


def find_images(input_dir: Path) -> list[Path]:
    """Return all JPG and JPEG files in alphabetical order."""
    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input folder not found: {input_dir}\n"
            "Create an 'input' folder next to the Python file and place the JPG images inside."
        )

    images = sorted(
        [
            *input_dir.glob("*.jpg"),
            *input_dir.glob("*.jpeg"),
            *input_dir.glob("*.JPG"),
            *input_dir.glob("*.JPEG"),
        ]
    )

    if not images:
        raise FileNotFoundError(f"No JPG images found in: {input_dir}")

    if len(images) != EXPECTED_IMAGE_COUNT:
        print(
            f"Warning: Expected {EXPECTED_IMAGE_COUNT} images, "
            f"but found {len(images)}. The script will continue."
        )

    return images


def benchmark_model(
    model_label: str,
    model_path: str,
    image_paths: list[Path],
) -> dict[str, Any]:
    """
    Benchmark one model.

    The measured Ultralytics inference time excludes preprocessing and
    postprocessing. Each image is processed individually with batch size 1.
    """
    print(f"\nLoading {model_label}: {model_path}")
    model = YOLO(model_path)

    print(f"Running {WARMUP_RUNS} warm-up passes...")
    warmup_image = str(image_paths[0])

    for _ in range(WARMUP_RUNS):
        model.predict(
            source=warmup_image,
            imgsz=IMAGE_SIZE,
            device=DEVICE,
            verbose=False,
        )

    per_image_rows: list[dict[str, Any]] = []
    all_inference_times: list[float] = []
    all_preprocess_times: list[float] = []
    all_postprocess_times: list[float] = []

    print(f"Benchmarking {len(image_paths)} images...")

    for image_path in image_paths:
        image_inference_times: list[float] = []
        image_preprocess_times: list[float] = []
        image_postprocess_times: list[float] = []

        for _ in range(REPEATS_PER_IMAGE):
            results = model.predict(
                source=str(image_path),
                imgsz=IMAGE_SIZE,
                device=DEVICE,
                verbose=False,
            )

            speed = results[0].speed

            image_preprocess_times.append(float(speed["preprocess"]))
            image_inference_times.append(float(speed["inference"]))
            image_postprocess_times.append(float(speed["postprocess"]))

        mean_preprocess = statistics.mean(image_preprocess_times)
        mean_inference = statistics.mean(image_inference_times)
        mean_postprocess = statistics.mean(image_postprocess_times)

        all_preprocess_times.append(mean_preprocess)
        all_inference_times.append(mean_inference)
        all_postprocess_times.append(mean_postprocess)

        per_image_rows.append(
            {
                "model": model_label,
                "image": image_path.name,
                "preprocess_ms": round(mean_preprocess, 3),
                "inference_ms": round(mean_inference, 3),
                "postprocess_ms": round(mean_postprocess, 3),
            }
        )

        print(f"  {image_path.name}: {mean_inference:.2f} ms inference")

    average_inference = statistics.mean(all_inference_times)
    standard_deviation = (
        statistics.stdev(all_inference_times)
        if len(all_inference_times) > 1
        else 0.0
    )
    fps = 1000.0 / average_inference if average_inference > 0 else 0.0

    return {
        "model": model_label,
        "average_preprocess_ms": statistics.mean(all_preprocess_times),
        "average_inference_ms": average_inference,
        "average_postprocess_ms": statistics.mean(all_postprocess_times),
        "standard_deviation_ms": standard_deviation,
        "fps": fps,
        "per_image_rows": per_image_rows,
    }


def save_csv(results: list[dict[str, Any]], output_path: Path) -> None:
    """Save all per-image measurements to a CSV file."""
    rows = []

    for result in results:
        rows.extend(result["per_image_rows"])

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "model",
                "image",
                "preprocess_ms",
                "inference_ms",
                "postprocess_ms",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def save_chart(results: list[dict[str, Any]], output_path: Path) -> None:
    """Create a compact bar chart comparing average inference times."""
    model_names = [result["model"] for result in results]
    average_times = [result["average_inference_ms"] for result in results]
    standard_deviations = [result["standard_deviation_ms"] for result in results]

    fig, ax = plt.subplots(figsize=(6.2, 4.0))

    bars = ax.bar(
        model_names,
        average_times,
        yerr=standard_deviations,
        capsize=5,
    )

    ax.set_title("CPU inference time comparison")
    ax.set_ylabel("Average inference time [ms / image]")
    ax.set_xlabel("Pose model")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, value in zip(bars, average_times):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f} ms",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


# ------------------------------------------------------------
# Main program
# ------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = find_images(INPUT_DIR)

    cpu_name = get_cpu_name()
    ram_gb = get_ram_gb()

    print("=" * 64)
    print("YOLO pose inference benchmark")
    print("=" * 64)
    print(f"CPU:          {cpu_name}")
    print(f"RAM:          {ram_gb:.1f} GB" if ram_gb is not None else "RAM:          Unknown")
    print(f"Device:       {DEVICE}")
    print(f"Input size:   {IMAGE_SIZE} x {IMAGE_SIZE}")
    print(f"Images:       {len(image_paths)}")
    print(f"Warm-up runs: {WARMUP_RUNS}")
    print(f"Repeats:      {REPEATS_PER_IMAGE} per image")

    benchmark_results = []

    for model_label, model_path in MODELS.items():
        result = benchmark_model(
            model_label=model_label,
            model_path=model_path,
            image_paths=image_paths,
        )
        benchmark_results.append(result)

    print("\n" + "=" * 64)
    print("Average results")
    print("=" * 64)

    for result in benchmark_results:
        print(
            f"{result['model']}: "
            f"{result['average_inference_ms']:.2f} ms/image "
            f"± {result['standard_deviation_ms']:.2f} ms, "
            f"{result['fps']:.2f} FPS"
        )

    csv_path = OUTPUT_DIR / "pose_inference_times.csv"
    chart_path = OUTPUT_DIR / "pose_inference_comparison.png"

    save_csv(benchmark_results, csv_path)
    save_chart(benchmark_results, chart_path)

    print("\nSaved files:")
    print(f"  CSV:   {csv_path}")
    print(f"  Chart: {chart_path}")


if __name__ == "__main__":
    main()