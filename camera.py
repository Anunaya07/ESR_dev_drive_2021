#imported necessary libraries
import cv2
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
from deepface import DeepFace # import DeepFace

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
classifier =load_model(r'./models/emotion_detection_model.h5') #emotion model loaded
signModel=load_model(r'./models/sign_lang_recog_train_model5.h5') #sign model loaded
faceCascade= cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') #haarcascade used for face recognisation

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise'] #emotion list




class Video_Emotion_1(object):
    def __init__(self):
         self.cap = cv2.VideoCapture(0) #camera acessing
    def __del__(self):
       self.cap.release()
    def get_frame(self):
        _, frame = self.cap.read()
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
           _,jpg=cv2.imencode(".jpg",frame)
           return jpg.tobytes()


class Video_Emotion_2(object):
    def __init__(self):
         self.cap = cv2.VideoCapture(0)#camera acessing
         cv2.destroyAllWindows()
    def __del__(self):
       self.cap.release()
    def get_frame(self):
        _, frame = self.cap.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #conversion to gray scale as the model is trained in gray scale
        faces = faceCascade.detectMultiScale(gray) #face demensions
        try:
                for (x,y,w,h) in faces:
                    cv2.rectangle(frame,(x,y),(x+w, y+h), (0,255,0), 2) #drawing rectangle around face
                    result = DeepFace.analyze(frame, actions = ['emotion'])
                    label= result['dominant_emotion']
                    label_position=(x,y-20)
                    font = cv2.FONT_HERSHEY_SIMPLEX #chosing font
                    cv2.putText(frame,label,label_position, font, 1, (0,0,255),2) #writing text
                _,jpg=cv2.imencode(".jpg",frame)
                return jpg.tobytes()
        except:
                return "I can't detect your beautiful face! Sorry:-(!"



class Video_Sign(object):
    def __init__(self):
         self.cap = cv2.VideoCapture(0) #camera acessing
    def __del__(self):
       self.cap.release()
    
    def get_frame(self):
        def letter(r):
            dic={0: 'Blank',1: 'A',2: 'B',3: 'C',4: 'D',5: 'E',6: 'F',7: 'G',8: 'H',9: 'I',10: 'J',11: 'K',12: 'L',13: 'M',14: 'N',15: 'O',16: 'P',17: 'Q',18: 'R',19: 'S',20: 'T',21: 'U',22: 'V',23: 'W',24: 'X',25: 'Y',26: 'Z'}
            return dic[r]
        _, frame = self.cap.read()
    # Simulating mirror image
        frame = cv2.flip(frame, 1)
        x1 = int(0.5*frame.shape[1])
        y1 = 10
        x2 = frame.shape[1]-10
        y2 = int(0.5*frame.shape[1])
        cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,2)
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        roi = cv2image[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,(5,5),2)
        th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
        ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        test_image=cv2.resize(res,(128,128))
        result=signModel.predict(test_image.reshape(1, 128, 128, 1))
        rounded_predictions=np.argmax(result,axis=-1)
        cv2.putText(frame,str(letter(rounded_predictions[0])),(80,100),cv2.FONT_HERSHEY_COMPLEX,2,(0,255,0),2)
        cv2.putText(frame,"Make gestures here",(363,360),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,255,0),2)
        _,jpg=cv2.imencode(".jpg",frame)
        return jpg.tobytes()


