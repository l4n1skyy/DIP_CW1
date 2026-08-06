import cv2

face_cascade = cv2.CascadeClassifier("face_detector.xml")


def blur_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for x, y, w, h in faces:
        frame[y : y + h, x : x + w] = cv2.GaussianBlur(
            frame[y : y + h, x : x + w], (51, 51), 30
        )
    return frame
