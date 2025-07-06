import cv2
import os

# Ask user for name
name = input("Enter your name: ")
folder_path = f"TrainingImages/{name}"
os.makedirs(folder_path, exist_ok=True)

# Load Haar Cascade face detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

img_id = 0

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        img_id += 1
        face = gray[y:y+h, x:x+w]
        cv2.imwrite(f"{folder_path}/{name}_{img_id}.jpg", face)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('Capturing Faces - Press Enter to Stop', frame)

    if cv2.waitKey(1) == 13 or img_id >= 20:
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Face collection completed.")
