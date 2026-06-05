from ultralytics import YOLO
import cv2
import torch
import os

# Modell laden
model = YOLO("yolo26n-pose.pt")

raw = model(embed=[-1])

# GPU prüfen
device = 0 if torch.cuda.is_available() else "cpu"
print("Running on:", "GPU" if device == 0 else "CPU")

# Bildpfad
image_path = os.path.join(
    "Beispielbilder",
    "lhccoutinho-soccer-4275827_1280.jpg"
)

# Vorhersage
results = model.predict(source=image_path, device=device)

# Ergebnis holen
r = results[0]

r = results[0]

print("=== Results Object Analysis ===")
print("Type of r:", type(r))
print("Available attributes:", [attr for attr in dir(r) if not attr.startswith('_')])

frame = r.plot()

frame = cv2.resize(frame, (600, 900))

# Anzeigen
cv2.imshow("YOLO26 Pose", frame)

# Warten bis Taste gedrückt wird
cv2.waitKey(0)

cv2.destroyAllWindows()