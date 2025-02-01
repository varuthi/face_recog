import face_recognition
import cv2
import numpy as np

# Load known images and encode them
known_image = face_recognition.load_image_file("varun.jpeg")  
known_encoding = face_recognition.face_encodings(known_image)[0]

jude_image = face_recognition.load_image_file("jude.jpg")  
jude_encoding = face_recognition.face_encodings(jude_image)[0]

naveen_image = face_recognition.load_image_file("naveen.jpg")  
naveen_encoding = face_recognition.face_encodings(naveen_image)[0]

# Store encodings with names
known_face_encodings = [
    known_encoding,
    jude_encoding,
    naveen_encoding]
known_face_names = [
    "Varun",
    "jude",
    "Naveen"
    ]



# Initialize webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    
    # Resize frame to 1/4 size for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces and encode them
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []

    for face_encoding in face_encodings:
        # Compare face encodings
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
        name = "Unknown"

        # Use the first match found
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        face_names.append(name)

    # Draw rectangles and names
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the resulting frame
    cv2.imshow("Face Recognition", frame)

    # Quit with 'q' key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release webcam and close windows
video_capture.release()
cv2.destroyAllWindows()
