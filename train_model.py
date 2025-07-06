import cv2
import os
import numpy as np
import pickle

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
x_train = []
y_labels = []

for root, dirs, files in os.walk("TrainingImages"):
    for file in files:
        if file.endswith("jpg") or file.endswith("png"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "_").lower()

            if label not in label_ids:
                label_ids[label] = current_id
                current_id += 1

            id_ = label_ids[label]
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            faces = face_cascade.detectMultiScale(img, 1.3, 5)
            for (x, y, w, h) in faces:
                roi = img[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)

# Save label names
with open("Trainer/labels.pickle", "wb") as f:
    pickle.dump(label_ids, f)

# Train the model and save it
recognizer.train(x_train, np.array(y_labels))
recognizer.save("Trainer/trainer.yml")

print("✅ Model trained and saved to Trainer/trainer.yml")
