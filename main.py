import cv2
import numpy
import os

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')

cap = cv2.VideoCapture(0)
k=0
running=True
while(running):
    _,frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = frame[y:y+h,x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_color)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey),(ex+ew,ey+eh),(0,0,255),1)
            eye_roi= roi_gray[ey:ey+eh,ex:ex+ew]
            if(k%3==0):
                image_name= f'data/more left/{str(k)}.jpg'  #"more right" and "more left" to increase the dataset
                cv2.imwrite(image_name,eye_roi)
            k=k+1

    cv2.imshow('yes',frame)
    if (cv2.waitKey(1)==ord('q')):
        running =False

