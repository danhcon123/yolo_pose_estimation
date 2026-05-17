from ultralytics import YOLO
import cv2, torch


model = YOLO("yolo26n-pose.pt")
device = 0 if torch.cuda.is_available() else "cpu"

print("Running on:", "GPU" if device == 0 else "CPU")

result = model.predict(source="dance_2.mp4", stream=True, device=device)
print(f"Result: {result}")

for r in result:
    frame = r.plot()
    frame = cv2.resize(frame, (960, 540))  # change display size here
    cv2.imshow("YOLO26 Pose", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()