import cv2
import os
import numpy as np

# Load the known face data from the "dataset" folder
known_face_images = []
known_face_labels = []

# Path to the "dataset" folder containing images of different people
dataset_dir = 'dataset'

# Create a list of subdirectories in the "dataset" folder, where each subdirectory represents a person
people = os.listdir(dataset_dir)

# Assign labels to each person
label = 0
for person in people:
    person_path = os.path.join(dataset_dir, person)
    if os.path.isdir(person_path):
        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            known_face_images.append(image)
            known_face_labels.append(label)
        label += 1

# Initialize the recognizer with LBPHFaceRecognizer
face_recognizer = cv2.face_LBPHFaceRecognizer.create()

# Train the recognizer with the known faces
face_recognizer.train(known_face_images, np.array(known_face_labels))

# Initialize the camera
camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()

    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Use a face detection classifier (e.g., Haar Cascade Classifier)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Crop the detected face
        face = gray_frame[y:y+h, x:x+w]

        # Perform face recognition on the detected face
        label, confidence = face_recognizer.predict(face)

        if confidence < 100:  # Adjust the confidence threshold as needed
            name = people[label]
        else:
            name = "Unknown"

        # Draw a rectangle around the detected face and display the name
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, str(name), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Face Recognition", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close OpenCV window
camera.release()
cv2.destroyAllWindows()
