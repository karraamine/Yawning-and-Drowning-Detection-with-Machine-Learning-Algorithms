import cv2
from keras.models import load_model
import numpy as np 
from time import sleep
from pygame import mixer
import dlib 
from imutils import face_utils

def get_best_eyes_pair(eyes):
        min_dist = 80
        max_dist = 150
        eye_pairs = [] 
        for i in range(len(eyes)):
            for j in range(i+1, len(eyes)):
                eye1 = eyes[i]
                eye2 = eyes[j]       
                 
                # c1 cordinate of eye 1 center  ,c2 cordinate of eye2 center
                c1x = eye1[0] + eye1[2]/2
                c1y = eye1[1] + eye1[3]/2
                c2x = eye2[0] + eye2[2]/2
                c2y = eye2[1] + eye2[3]/2
                distance = euclidean_distance(c1x,c1y,c2x,c2y)
                if min_dist < distance < max_dist :
                    eye_pairs.append((eye1, eye2))   
        best_pair = []
        if len(eye_pairs) > 0:
            best_pair = min(eye_pairs,key=lambda pair: np.abs(pair[0][0] - pair[1][0]))
        return best_pair

def euclidean_distance(x1,y1,x2,y2):
    return np.sqrt( (x1 - x2)**2 + (y1 - y2)**2 )


def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance


print('[INFO] loading haar cascades ... ')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

font = cv2.FONT_HERSHEY_SIMPLEX


print('[INFO] loading the model')
model  = load_model('my_model.h5')
frames_threshold = 50
frames_threshold2= 90
counter = 0
is_open = True
is_yawn=False
print('[INFO] loading pygame ')
mixer.init()
sound = mixer.Sound('alarm.wav')

print('[INFO] initialize camera 0 ...')
cap = cv2.VideoCapture(0)

#skip = 5
prediction_buffer = []
prediction_buffer_size = 10
thicc=2
YAWN_THRESH = 20



sleep(2)
while True:
    _ ,frame = cap.read()
    
    #for _ in range(skip - 1):
    #    cap.read()
    height,width = frame.shape[:2] 
    cv2.rectangle(frame, (0,0) , (300,60) , (0,0,0) , thickness=cv2.FILLED )
    cv2.rectangle(frame, (width-170,0) , (width,60) , (0,0,0) , thickness=cv2.FILLED )
    
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(300, 300),flags=cv2.CASCADE_SCALE_IMAGE) 

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi_gray  = gray[y:y+int(2*h/3), x:x+w]
        roi_color = frame[y:y+h,x:x+w]        
        #mouph detection 
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        distance = lip_distance(shape)
        lip = shape[48:60]
        cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)
        
        if distance > YAWN_THRESH:
            is_yawn=True
            
        else :
            is_yawn=False
                
        
        
        
        
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 4,minSize=(65,65), maxSize=(85,85))
        best_pair = get_best_eyes_pair(eyes)
        for (ex,ey,ew,eh) in best_pair:
            cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 0, 255), 2)
           
            # eyes region of interest 
            eye_roi     = roi_color[ey:ey+eh,ex:ex+ew]
            resized_img = cv2.resize(eye_roi, (224,224))
            resized_img = np.expand_dims(resized_img, axis=0)
            resized_img = resized_img / 255.0
            
            prediction = model.predict(resized_img)
            prediction_buffer.append(prediction)
            prediction_buffer = prediction_buffer[-prediction_buffer_size:]
            average_prediction = np.mean(prediction_buffer, axis=0)
           # prediction = model.predict(resized_img)
            x1,y1,w1,h1 = 0,0,175,30
            
            #if prediction > 0.5 :
            if average_prediction > 0.5 :
                is_open = True
                cv2.putText(frame,"Open",(10,40), font, 1,(255,255,255),1,cv2.LINE_AA)
            else : 
                is_open = False
                cv2.putText(frame,"Closed",(10,40), font, 1,(255,255,255),1,cv2.LINE_AA)
    
    if not is_open :
        counter += 1
    else : 
        if counter > 0 :
            counter -= 1
    if is_yawn :
      cv2.putText(frame, "Yawn Alert", (width - 150, 35),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255, 255), 2)  
    else :
        cv2.putText(frame, "NO Yawn ", (width - 150, 35),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if (counter > frames_threshold and  is_yawn ) or counter > frames_threshold2 : 
        try:          
            sound.play()
            if(thicc<16):
             thicc= thicc+2
            else:
              thicc=thicc-2
              if(thicc<2):
                thicc=2
            cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc) #BGR
        except:  # isplaying = False
            pass
        
    score = int(counter/30)
    cv2.putText(frame,'Score:'+str(score),(150,40), font, 1,(255,255,255),1,cv2.LINE_AA)
    #print(f'counter = {counter}')    

    cv2.imshow('EYE DETECTION', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

