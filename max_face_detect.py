import cv2
import time
import mediapipe as mp

# Initializing MediaPipe Face Mesh solution
max_faces_to_test = [1, 5, 10, 15, 20, 25]  # List of max faces to test

# Initializing the drawing utils for drawing the facial landmarks on image
mp_drawing = mp.solutions.drawing_utils
capture = cv2.VideoCapture(0)

# Initializing current time and previous time for calculating the FPS
previousTime = time.time()

while capture.isOpened():
    # Capture frame by frame
    ret, frame = capture.read()

    # Check if frame is read correctly
    if not ret:
        print("Failed to capture image")
        break

    for max_faces in max_faces_to_test:
        # Initializing MediaPipe Face Mesh solution with dynamic max_num_faces
        face_mesh_model = mp.solutions.face_mesh.FaceMesh(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_faces=max_faces
        )

        # Converting the BGR frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Making predictions using the Face Mesh model
        frame_rgb.flags.writeable = False
        results = face_mesh_model.process(frame_rgb)
        frame_rgb.flags.writeable = True

        # Converting back to BGR for displaying
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Drawing the facial landmarks for each face detected
        if results.multi_face_landmarks:
            face_count = 0
            for face_landmarks in results.multi_face_landmarks:
                face_count += 1
                mp_drawing.draw_landmarks(
                    frame_bgr,
                    face_landmarks,
                    mp.solutions.face_mesh.FACEMESH_CONTOURS,
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
            cv2.putText(frame_bgr, f'{face_count}/{max_faces} faces detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            # Display no faces detected
            cv2.putText(frame_bgr, 'No faces detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Calculating the FPS
        currentTime = time.time()
        elapsedTime = currentTime - previousTime
        if elapsedTime > 0:  # Prevent division by zero
            fps = 1 / elapsedTime
        else:
            fps = 0
        previousTime = currentTime

        # Displaying FPS on the image
        cv2.putText(frame_bgr, f'{int(fps)} FPS', (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow(f"Max Faces: {max_faces}", frame_bgr)

    # Enter key 'q' to break the loop
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# When all the processes are done
# Release the capture and destroy all windows
capture.release()
cv2.destroyAllWindows()
