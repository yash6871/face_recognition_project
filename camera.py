import cv2

cam = cv2.VideoCapture(0)  # 0 = default webcam

while True:
    ret, frame = cam.read()
    if not ret:
        break

    cv2.imshow("Camera", frame)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cam.release()
cv2.destroyAllWindows()
