import cv2
import os

dataset_path = "dataset"
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

cam = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

user_id = input("Enter User ID (integer): ")
user_name = input("Enter User Name: ")

# Save name mapping
import numpy as np
import os
names_file = "names.npy"
if os.path.exists(names_file):
    names = list(np.load(names_file, allow_pickle=True))
else:
    names = ["Unknown"]

# Extend list to match ID index
while len(names) <= int(user_id):
    names.append("Unknown")

names[int(user_id)] = user_name
np.save(names_file, names)

count = 0

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        cv2.imwrite(f"{dataset_path}/User.{user_id}.{count}.jpg", gray[y:y+h, x:x+w])
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('Collecting Faces', img)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break
    elif count >= 50:  # Collect 50 samples
        break

cam.release()
cv2.destroyAllWindows()
print("✅ Dataset collection complete.")
