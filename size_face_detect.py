import cv2
import time
import mediapipe as mp

# Initializing MediaPipe Face Mesh solution
mp_face_mesh = mp.solutions.face_mesh
face_mesh_model = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_faces=5
)

# Initializing the drawing utils for drawing the facial landmarks on image
mp_drawing = mp.solutions.drawing_utils
capture = cv2.VideoCapture(0)

# Initializing current time and previous time for calculating the FPS
previousTime = time.time()

# Define the range of sizes to test
test_sizes = [50, 100, 200, 300, 400, 500, 600]  # in pixels

while capture.isOpened():
    # Capture frame by frame
    ret, frame = capture.read()

    # Check if frame is read correctly
    if not ret:
        print("Failed to capture image")
        break

    for size in test_sizes:
        # Resizing the frame to different sizes for testing
        resized_frame = cv2.resize(frame, (size, size))

        # Converting the BGR frame to RGB
        resized_frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

        # Making predictions using the Face Mesh model
        resized_frame_rgb.flags.writeable = False
        results = face_mesh_model.process(resized_frame_rgb)
        resized_frame_rgb.flags.writeable = True

        # Drawing the facial landmarks for each face detected
        if results.multi_face_landmarks:
            face_count = 0
            for face_landmarks in results.multi_face_landmarks:
                face_count += 1
                mp_drawing.draw_landmarks(
                    resized_frame,
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
            cv2.putText(resized_frame, f'{face_count} faces detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            # Display no faces detected
            cv2.putText(resized_frame, 'No faces detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Calculating the FPS
        currentTime = time.time()
        elapsedTime = currentTime - previousTime
        if elapsedTime > 0:  # Prevent division by zero
            fps = 1 / elapsedTime
        else:
            fps = 0
        previousTime = currentTime

        # Displaying FPS on the image
        cv2.putText(resized_frame, f'{int(fps)} FPS', (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        # Display the resulting resized frame
        cv2.imshow(f"Size: {size}px", resized_frame)

    # Enter key 'q' to break the loop
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# When all the processes are done
# Release the capture and destroy all windows
capture.release()
cv2.destroyAllWindows()
