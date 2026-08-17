import cv2

face_cascade = cv2.CascadeClassifier("face_detector.xml")


def blur_faces(frame):
    #load the frame as gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #equalize the histogram to perform contrast stretching
    gray = cv2.equalizeHist(gray)
    #detect faces using the cascadeclassifier with scaleFactor 1.1 
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4,minSize=(40,40))
    for x, y, w, h in faces:
        frame[y : y + h, x : x + w] = cv2.GaussianBlur(
            frame[y : y + h, x : x + w], (51, 51), 30
        )
    return frame
