import cv2
import numpy as np
from PIL import Image
import os

dataset_path = "dataset"
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    face_samples = []
    ids = []
    for imagePath in image_paths:
        pil_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(pil_img, 'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)
        for (x, y, w, h) in faces:
            face_samples.append(img_numpy[y:y+h, x:x+w])
            ids.append(id)
    return face_samples, ids

faces, ids = get_images_and_labels(dataset_path)
recognizer.train(faces, np.array(ids))
recognizer.write("trainer.yml")

print("✅ Training complete. Model saved as trainer.yml")
