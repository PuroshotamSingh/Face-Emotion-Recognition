'''
This python file contains code to demonstrate real time face emotion detection using webcam
'''

# importing relevant libraries
import cv2
import numpy as np
from time import sleep
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
# loading the necessary files
face_classifier = cv2.CascadeClassifier(r'C:\Users\Gauri Singh\Desktop\Deep Learning projects\Face_Emotion_detection_Puroshotam\haarcascade_frontalface_default.xml')
classifier =load_model(r'C:\Users\Gauri Singh\Desktop\Deep Learning projects\Face_Emotion_detection_Puroshotam\model.h5')

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

cap = cv2.VideoCapture(0)                                                       # capturing the video live webcam



while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)                               # converting image into grayscale image
    faces = face_classifier.detectMultiScale(gray)                              # (region of interest of detected face)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)                      # draw rectangle over each face
        roi_gray = gray[y:y+h,x:x+w]                                            # croping gray scale image
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)    # image is resized to 48,48

        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0                                # feature scaling of final image
            roi = img_to_array(roi)                                             # converting to numpy array
            roi = np.expand_dims(roi,axis=0)                                    # array is expanded by inserting axis at position 0

            prediction = classifier.predict(roi)[0]                             # predicting emotion of captured image from the trained model
            label=emotion_labels[prediction.argmax()]                           # finding the label of class which has maximaum probalility
            label_position = (x,y-10)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)  # putText is used to draw a detected emotion on image
            cv2.putText(frame,'Press q to exit',(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
        else:
            cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.imshow('Emotion Detector',frame)                                        # showing result
    if cv2.waitKey(1) & 0xFF == ord('q'):                                       # press q to exit
        break

cap.release()
cv2.destroyAllWindows()