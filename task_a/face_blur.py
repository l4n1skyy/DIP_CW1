import cv2

#Load the face dector xml file
face_cascade = cv2.CascadeClassifier("face_detector.xml")

def blur_faces(frame):
    #load the frame as gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #equalize the histogram to perform contrast stretching
    gray = cv2.equalizeHist(gray)
    #detect faces using the cascadeclassifier loaded
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    #get the x (how far from the left of the face starts)
    #get the y (how far from the top of the face starts)
    #get the w (how wide the face is)
    #get the h (how tall the face is)
    for x, y, width, height in faces:
        #Apply GaussianBlur on the pixels the face is on with a 51,51 kernel and sigma 30
        frame[y : y + height, x : x + width] = cv2.GaussianBlur(frame[y : y + height, x : x + width], (51, 51), 30)
    return frame
