import numpy as np
import serial
import cv2
import threading  # :C


global g_image  # :C


def draw_face(image, face):
    x, y, w, h = face
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return image


def cut_face(image, face):
    x, y, w, h = face
    return image[y:y + h, x:x + w]


serial_port = serial.Serial('/dev/ttyUSB0')


def handle_data(data):
    print(data)
    cv2.imshow('hi', g_image)


def read_from_port(ser):
    while True:
        print("test")
        reading = ser.readline().decode()
        handle_data(reading)


thread = threading.Thread(target=read_from_port, args=(serial_port,))
thread.start()


face_cascade = cv2.CascadeClassifier('frontalface_cascade.xml')
cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    ret, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    for face in faces:
        draw_face(image, face)

    cv2.imshow('frame', image)

    cut_faces = [cut_face(image, face) for face in faces]
    if len(cut_faces) != 0:
        g_image = cut_faces[0]
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
