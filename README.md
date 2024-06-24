# Multi-Face Detection with MediaPipe and OpenCV

This project uses MediaPipe Face Mesh and OpenCV to detect multiple faces in real-time from a webcam feed. It demonstrates how to adjust the maximum number of faces detected and test the performance of the model.

## Features

- Detects multiple faces in real-time using MediaPipe Face Mesh.
- Displays facial landmarks on each detected face.
- Calculates and displays frames per second (FPS).
- Adjustable maximum number of faces to detect.
- Resizes frames to test detection performance at different resolutions.

## Installation

### Prerequisites

- Python 3.7 or later
- OpenCV
- MediaPipe
- NumPy

### Install Dependencies

You can install the required libraries using pip:

```bash
pip install opencv-python-headless mediapipe numpy
```

## Usage
1. Clone this repository or download the script.
2. Run the script:
```bash
python face_landmark.py
```

## Testing
### Testing Maximum Number of Faces Detected
- The script includes a loop to test different values for the maximum number of faces detected. The default values tested are 1, 5, 10, 15, 20, and 25. You can modify the max_faces_to_test list to include any other values you want to test.

### Testing Minimum Face Size
- To test the minimum face size that the model can detect, you can resize the input frames to various smaller dimensions and observe the detection performance.

## License