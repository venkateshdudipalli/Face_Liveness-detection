

from flask import Flask, render_template, Response
import cv2
#from tensorflow.keras.preprocessing.image import img_to_array
import os
import numpy as np
from tensorflow.keras.models import model_from_json
root_dir = os.getcwd()
# Load Face Detection Model
#face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
# Load Anti-Spoofing Model graph
json_file = open('antispoofing_models/antispoofing_model.json','r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load antispoofing model weights 
model.load_weights('antispoofing_models/antispoofing_model.h5')
print("Model loaded from disk")
# 
app=Flask(__name__)
camera = cv2.VideoCapture(0)


def gen_frames():  
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
            #eye_cascade = cv2.CascadeClassifier('Haarcascades/haarcascade_eye.xml')
            try:
                faces=face_cascade.detectMultiScale(frame,1.1,7)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                 #Draw the rectangle around each face
                for (x, y, w, h) in faces:
                    face = frame[y-5:y+h+5,x-5:x+w+5]
                    resized_face = cv2.resize(face,(160,160))
                    resized_face = resized_face.astype("float") / 255.0
                    resized_face = np.expand_dims(resized_face, axis=0)
                    preds = model.predict(resized_face)[0]
                    print(preds)
                    if preds> 0.5:
                        label = 'spoof'
                        cv2.putText(frame, label, (x,y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                        cv2.rectangle(frame, (x, y), (x+w,y+h),(0, 0, 255), 2)
                    else:
                        label = 'real'
                        cv2.putText(frame, label, (x,y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                        cv2.rectangle(frame, (x, y), (x+w,y+h),(0, 255, 0), 2)
            except:
                print("No face found")
                
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__=='__main__':
    app.run(debug=True)
