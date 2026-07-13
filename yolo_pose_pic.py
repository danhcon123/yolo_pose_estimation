from ultralytics import YOLO
import cv2, torch

"""
# model = YOLO("yolo26l-pose.pt")
model = YOLO("yolo26n-pose.pt")
device = 0 if torch.cuda.is_available() else "cpu"

print("Running on:", "GPU" if device == 0 else "CPU")
source="input/running_man.jpg" # Image input path

result = model.predict(source, device=device, save=True)
result[0].show()  # Display the image with detections
print(f"Result: {result}")
"""

model = YOLO("yolo26n-pose.pt")

results = model("C:/Users/gauva/Desktop/Oberseminar/docu/pexels-ketut-subiyanto-4853109.jpg")

for r in results:
    img = r.plot(
        kpt_radius=40,   # bigger keypoints
        kpt_line=True,   # draw skeleton lines
        line_width= 30    # thicker box/skeleton drawing
    )

    cv2.imwrite("output_big_keypoints.jpg", img)