import cv2
import numpy as np
import pickle
import os
from datetime import datetime

# Load Haar Cascade and trained model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("Trainer/trainer.yml")

# Load label IDs (person names)
with open("Trainer/labels.pickle", "rb") as f:
    labels = pickle.load(f)
    labels = {v: k for k, v in labels.items()}

# Ensure Attendance folder exists
os.makedirs("Attendance", exist_ok=True)

# Create or open today's CSV file
today = datetime.now().strftime('%Y-%m-%d')
filename = f"Attendance/attendance_{today}.csv"

if not os.path.exists(filename):
    with open(filename, "w") as f:
        f.write("Name,Time\n")

# Function to mark attendance
def mark_attendance(name):
    with open(filename, "r+") as f:
        lines = f.readlines()
        names = [line.split(',')[0] for line in lines]
        if name not in names:
            now = datetime.now().strftime('%H:%M:%S')
            f.write(f"{name},{now}\n")
            print(f"✅ {name} marked present at {now}")

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        id_, conf = recognizer.predict(roi)

        if conf < 70:
            name = labels.get(id_, "Unknown")
            mark_attendance(name)
            color = (0, 255, 0)
        else:
            name = "Unknown"
            color = (0, 0, 255)

        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    cv2.imshow("Face Recognition - Press Q to Quit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Attendance session ended.")
