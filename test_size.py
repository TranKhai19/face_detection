import cv2
import mediapipe as mp

# Khởi tạo MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh_model = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_faces=10
)

# Khởi tạo drawing utils để vẽ các facial landmarks
mp_drawing = mp.solutions.drawing_utils

# Mở video capture với độ phân giải cao
capture = cv2.VideoCapture("video2.mp4")  # Hoặc thay bằng đường dẫn tới video của bạn
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Thiết lập độ phân giải khung hình
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        break

    # Chuyển đổi từ BGR sang RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Dùng mô hình MediaPipe Face Detection để phát hiện khuôn mặt
    results = face_mesh_model.process(frame_rgb)

    # Chuyển đổi lại từ RGB sang BGR để hiển thị
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    # Vẽ các facial landmarks nếu phát hiện được khuôn mặt
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                frame_bgr,
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

    # Hiển thị ảnh với các khuôn mặt đã được phát hiện và landmarks vẽ lên
    cv2.imshow('Face Mesh from Video', frame_bgr)

    # Nhấn 'q' để thoát
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Giải phóng bộ nhớ và đóng tất cả các cửa sổ
capture.release()
cv2.destroyAllWindows()
