import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# Load the pre-trained model from TensorFlow Hub
model_url = "https://www.kaggle.com/code/gauravsharma99/facial-emotion-recognition"
model = hub.load(model_url)

# Define the emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load the pre-trained face cascade for detecting faces
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open the webcam (you can also use a video file instead)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Process each detected face
    for (x, y, w, h) in faces:
        # Extract the face region
        face_roi = gray[y:y + h, x:x + w]

        # Resize the face image to match the input size of the model
        face_roi = cv2.resize(face_roi, (224, 224))  # Adjust based on the model's input size
        face_roi = np.expand_dims(face_roi, axis=-1)
        face_roi = np.expand_dims(face_roi, axis=0)
        face_roi = face_roi / 255.0  # Normalize the pixel values

        # Predict your emotions
        emotion_probabilities = model.predict(face_roi)
        emotion_index = np.argmax(emotion_probabilities)
        emotion = emotion_labels[emotion_index]

        # Draw a rectangle around the detected face and display the predicted emotion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, f'Emotion: {emotion}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Emotion Recognition', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
