import cv2
from keras.models import load_model
import os
import numpy as np
import pandas as pd
model=load_model('best_model.keras')

video=cv2.VideoCapture(0)

label_dict = {0 : 'surprise', 1 : 'happy', 2 : 'anger', 3 : 'disgust', 4 : 'fear', 5 : 'sad', 6 : 'neutral'}

faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# labels_dict={0:'Angry',1:'Disgust', 2:'Fear', 3:'Happy',4:'Neutral',5:'Sad',6:'Surprise'}

while True:
    ret, frame = video.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert the frame to grayscale and detect faces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 3)

    for x, y, w, h in faces:
        # Extract the face region
        sub_face_img = frame[y:y+h, x:x+w]  # Use the color frame
        resized = cv2.resize(sub_face_img, (96, 96))  # Resize to match the model's input
        normalized = resized / 255.0  # Normalize pixel values
        reshaped = np.reshape(normalized, (1, 96, 96, 3))  # Reshape for model input

        # Make a prediction
        result = model.predict(reshaped)
        label = np.argmax(result, axis=1)[0]

        # Draw bounding box and label on the frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
        cv2.putText(frame, label_dict[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Display the frame
    cv2.imshow('Emotion Recognition', frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

# Release resources
video.release()
cv2.destroyAllWindows()