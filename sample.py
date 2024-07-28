from twilio.rest import Client
import face_recognition
import cv2

# Twilio credentials
account_sid = 'your_account_sid'
auth_token = 'your_auth_token'
twilio_phone_number = 'your_twilio_phone_number'
receiver_phone_number = 'receiver_phone_number'  # The phone number to receive the alert

# Load a sample picture and learn how to recognize it.
known_image = face_recognition.load_image_file("ok.jpg")
known_face_encoding = face_recognition.face_encodings(known_image)[0]

# Initialize the camera
video_capture = cv2.VideoCapture(0)

# Initialize door status
door_open = False

# Twilio client
client = Client(account_sid, auth_token)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces([known_face_encoding], face_encoding)

        if True in matches:  # If a known face is detected, open the door and send an alert
            door_open = True
            print("Door opened")

            # Send alert message
            message = client.messages.create(
                body='Someone is at the door!',
                from_=twilio_phone_number,
                to=receiver_phone_number
            )
            print("Alert message sent to the owner.")
            break

    # If door is open, simulate opening the door
    if door_open:
        # Your code to open the door goes here
        # For example, you can trigger a GPIO pin on a Raspberry Pi

        # Reset the door status
        door_open = False

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()