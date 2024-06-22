import cv2
import time
import mediapipe as mp

# Initializing MediaPipe Face Mesh solution
mp_face_mesh = mp.solutions.face_mesh
face_mesh_model = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_faces=5  # Increase this number to detect more faces
)

# Initializing the drawing utils for drawing the facial landmarks on image
mp_drawing = mp.solutions.drawing_utils
capture = cv2.VideoCapture(0)

# Initializing current time and previous time for calculating the FPS
previousTime = 0
currentTime = 0

while capture.isOpened():
    # Capture frame by frame
    ret, frame = capture.read()

    # Check if frame is read correctly
    if not ret:
        print("Failed to capture image")
        break

    # Resizing the frame for better view
    frame = cv2.resize(frame, (800, 600))

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Making predictions using the Face Mesh model
    # To improve performance, optionally mark the image as not writeable to pass by reference.
    image.flags.writeable = False
    results = face_mesh_model.process(image)
    image.flags.writeable = True

    # Converting back the RGB image to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Drawing the facial landmarks for each face detected
    if results.multi_face_landmarks:
        face_count = 0
        for face_landmarks in results.multi_face_landmarks:
            face_count += 1
            mp_drawing.draw_landmarks(
                image,
                face_landmarks,
                mp_face_mesh.FACEMESH_CONTOURS,
                mp_drawing.DrawingSpec(
                    color=(255, 0, 255),
                    thickness=1,
                    circle_radius=1
                ),
                mp_drawing.DrawingSpec(
                    color=(0, 255, 255),
                    thickness=1,
                    circle_radius=1
                )
            )
        
        # Display number of faces detected
        cv2.putText(image, f'{face_count} faces detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        # Display no faces detected
        cv2.putText(image, 'No faces detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Calculating the FPS
    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime

    # Displaying FPS on the image
    cv2.putText(image, f'{int(fps)} FPS', (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting image
    cv2.imshow("Multi-Face Landmarks", image)

    # Enter key 'q' to break the loop
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# When all the processes are done
# Release the capture and destroy all windows
capture.release()
cv2.destroyAllWindows()
