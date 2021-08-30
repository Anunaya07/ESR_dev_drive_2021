#imported necessary libraries
import cv2
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
classifier =load_model(r'emotion_detection_model.h5') #model loaded
emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise'] #emotion list




class Video(object):
    def __init__(self):
         self.cap = cv2.VideoCapture(0) #camera acessing
    def __del__(self):
       self.cap.release()
    def get_frame(self):
        _, frame = self.cap.read()
        if not _:
            print("Oops! looks like I don't have access to your camera :-(") #unble to access camera error message
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #conversion to gray scale as the model is trained in gray scale
        faces = face_classifier.detectMultiScale(gray)
        for (x,y,w,h) in faces:
           cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
           roi_gray = gray[y:y+h,x:x+w]
           roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)#resizing to the size in which the model is trained in.



           if np.sum([roi_gray])!=0:
              roi = roi_gray.astype('float')/255.0
              roi = img_to_array(roi) #converting image to array
              roi = np.expand_dims(roi,axis=0)

              prediction = classifier.predict(roi)[0]
              label=emotion_labels[prediction.argmax()]
              label_position = (x,y-20)
              cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
           else:
               cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
           cv2.imshow('Emotion Detector',frame)
           _,jpg=cv2.imencode(".jpg",frame)
           if cv2.waitKey(1) & 0xFF == ord('q'): #press q to exit the camera window!
                       break
           return jpg.tobytes()