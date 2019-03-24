import cv2 as cv
import numpy as np

#loading the haar cascade training file.
face_cascade = cv.CascadeClassifier('D:\My Code\Python\haarcascade_frontalface_default.xml')

#choosing our image file.
img = cv.imread('Face_Detection/NASA.jpg')

#converting the image to gray scale because cv2 only works with gray.
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)

#drawing a rectangle over every face detected.
for (x,y,w,h) in faces:
    cv.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

#print the number of detected faces.
print('Faces found:' , len(faces))

#display the image.
cv.imshow('img', img)

# a function to display the image for specified milliseconds.
#( waitKey(0) >> will display the image infinitely until any keypress).
k = cv.waitKey(0)
