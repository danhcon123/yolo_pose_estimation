from ultralytics import YOLO
import cv2
import torch
import os

def inspect_structure(obj, indent=0):
    prefix = "  " * indent

    if isinstance(obj, dict):
        print(f"{prefix}dict ({len(obj)} keys)")
        for k, v in obj.items():
            print(f"{prefix}  {k}:")
            inspect_structure(v, indent + 2)

    elif isinstance(obj, (list, tuple, set)):
        print(f"{prefix}{type(obj).__name__} ({len(obj)} items)")
        if obj:
            inspect_structure(next(iter(obj)), indent + 1)

    elif hasattr(obj, "__dict__"):
        print(f"{prefix}{type(obj).__name__}")
        for name, value in vars(obj).items():
            print(f"{prefix}  {name}:")
            inspect_structure(value, indent + 2)

    else:
        print(f"{prefix}{type(obj).__name__}")

# Modell laden
model = YOLO("yolo26n-pose.pt")

raw = model(embed=[-1])

# GPU prüfen
device = 0 if torch.cuda.is_available() else "cpu"
print("Running on:", "GPU" if device == 0 else "CPU")

# Bildpfad
image_path = os.path.join(
    "Beispielbilder",
    "jumping_woman.jpg"
)

# Vorhersage
results = model.predict(source=image_path, device=device)

# Ergebnis holen
r = results[0]

#print("raw")
#print(raw)
#inspect_structure(r)
print("--------------------------Keypoints werden hier gezeigt:-------------------------")
print(r.keypoints.has_visible)
#print("length:")
#print(len(raw))
# Bild mit Pose zeichnen
frame = r.plot()

# Rohdaten der Keypoints
#print("\n=== KEYPOINT DATA ===")
#inspect_structure(r)

'''# Nur x/y Koordinaten
print("\n=== XY ===")
print(r.keypoints.xy)

# Normalisierte Koordinaten (0-1)
print("\n=== XYN ===")
print(r.keypoints.xyn)

# Confidence Werte
print("\n=== CONF ===")
print(r.keypoints.conf)

# Bounding Boxes
print("\n=== BOXES ===")
print(r.boxes)

# Klassen
print("\n=== CLASSES ===")
print(r.boxes.cls)

# Confidence der Detection
print("\n=== BOX CONF ===")
print(r.boxes.conf)'''
# Größe ändern
frame = cv2.resize(frame, (600, 900))

# Anzeigen
cv2.imshow("YOLO26 Pose", frame)

# Warten bis Taste gedrückt wird
cv2.waitKey(0)

cv2.destroyAllWindows()