import cv2
import numpy as np
import mediapipe as mp

# Initializing MediaPipe Face Mesh solution
mp_face_mesh = mp.solutions.face_mesh
face_mesh_model = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_faces=10
)

# Initializing the drawing utils for drawing the facial landmarks on image
mp_drawing = mp.solutions.drawing_utils

# Create a blank image with white background
image = np.ones((800, 800, 3), dtype=np.uint8) * 255

# Draw circles of different sizes to simulate faces
face_sizes = [20, 30, 50, 70, 100, 150, 200, 250]
positions = [(100, 100), (250, 100), (400, 100), (550, 100), (700, 100), (250, 300), (400, 300), (550, 300)]

for size, pos in zip(face_sizes, positions):
    cv2.circle(image, pos, size, (0, 0, 0), -1)  # Draw black circles

# Convert BGR image to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process the image and find the faces
results = face_mesh_model.process(image_rgb)

# Convert back to BGR for drawing
image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

# Draw the face landmarks
if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        mp_drawing.draw_landmarks(
            image_bgr,
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

# Display the image with detected faces
cv2.imshow('Test Image', image_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
