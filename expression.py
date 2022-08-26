import numpy as np
import cv2
import tensorflow
import warnings
# from keras.preprocessing import image
warnings.filterwarnings("ignore")
from keras.models import model_from_json
# from keras.models import load_model

# model = load_model("model.h5")
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
model = model_from_json(open("model_structure.json", "r").read())
model.load_weights('model.h5') #load weights

emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

c=cv2.VideoCapture(0)
while True:
    ret, frame = c.read()  # captures frame and returns boolean value and captured image
    if not ret:
        continue
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) < 1:
        continue

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), thickness=2)
        d_face = frame[int(y):int(y+h), int(x):int(x+w)] #crop detected face
        d_face = cv2.cvtColor(d_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
        d_face = cv2.resize(d_face, (48, 48)) #resize to 48x48
        img_pixels = tensorflow.keras.preprocessing.image.img_to_array(d_face)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        img_pixels /= 255 #pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]
        predictions = model.predict(img_pixels) #store probabilities of 7 expressions

        #find max indexed array 0: angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral
        max_index = np.argmax(predictions[0])
        emotion = emotions[max_index]

    cv2.putText(frame, emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("face",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
        c.stop()
        break