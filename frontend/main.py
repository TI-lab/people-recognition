import numpy as np
import cv2


def draw_face(image, face):
    x, y, w, h = face
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return image


def cut_face(image, face):
    x, y, w, h = face
    return image[y:y + h, x:x + w]


face_cascade = cv2.CascadeClassifier('frontalface_cascade.xml')
cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    ret, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    for face in faces:
        draw_face(image, face)

    cv2.imshow('frame', image)

    cut_faces = [cut_face(image, face) for face in faces]
    if len(cut_faces) != 0:
        cv2.imshow('cut', cut_faces[0])
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
