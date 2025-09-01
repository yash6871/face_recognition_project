import cv2
import numpy as np

# Load trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

# Load name mappings
names = list(np.load("names.npy", allow_pickle=True))

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cam = cv2.VideoCapture(0)

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    for (x, y, w, h) in faces:
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        if confidence < 70:
            name = names[id]
            confidence_text = f"{round(100 - confidence)}%"
        else:
            name = "Unknown"
            confidence_text = f"{round(100 - confidence)}%"

        cv2.putText(img, str(name), (x+5, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(img, str(confidence_text), (x+5, y+h-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1)
        cv2.rectangle(img, (x, y), (x+w, y+h), (255,0,0), 2)

    cv2.imshow('Face Recognition', img)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cam.release()
cv2.destroyAllWindows()
