import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import time

from utils.model_loader import get_face_model_path
from utils.camera_setup import setup_camera

MODEL_NAME = "face_landmarker"

model_path = get_face_model_path(MODEL_NAME)

latest_result = None


def draw_landmarks_on_frame(
    frame: np.ndarray,
    face_landmarks_list: list,
    hide_frame = False
):
    if hide_frame:
        annotated_frame = np.ones(frame.shape, dtype=np.uint8)
    else:
        annotated_frame = frame.copy()
    
    for face_landmarks in face_landmarks_list:
        # Draw the face landmarks
        for landmark in face_landmarks:
            # Convert normalized coordinates to pixel coordinates
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            
            # Draw a small circle for each landmark
            cv2.circle(
                annotated_frame,
                (x, y),
                2,  # radius
                (0, 255, 0),  # color (green)
                -1  # filled circle
            )
            
        # Optionally draw connections between landmarks for better visualization
        connections = mp.solutions.face_mesh.FACEMESH_TESSELATION
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            
            start_point = face_landmarks[start_idx]
            end_point = face_landmarks[end_idx]
            
            x1 = int(start_point.x * frame.shape[1])
            y1 = int(start_point.y * frame.shape[0])
            x2 = int(end_point.x * frame.shape[1])
            y2 = int(end_point.y * frame.shape[0])
            
            cv2.line(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),  # color (green)
                1  # thickness
            )
    
    return annotated_frame

def callback(result: mp.tasks.vision.FaceLandmarkerResult, 
            output_image: mp.Image, 
            timestamp_ms: int):
    """Callback function to handle face landmark detection results."""
    global latest_result
    latest_result = result

def print_result(result: mp.tasks.vision.FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    print('face landmarker result: {}'.format(result))


def process_face_landmarks(cam, model_path, hide_frame=True):

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=callback)

    with FaceLandmarker.create_from_options(options) as landmarker:
        start_time = time.time()
        while True:
            ret, frame = cam.read()
            if not ret:
                raise IOError("Could not read camera data")

            # cv2.imshow("test",frame)

            current_time = time.time()
            timestamp_us = int((current_time - start_time) * 1_000_000)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            landmark_image = landmarker.detect_async(mp_image, timestamp_us)

            if latest_result is not None and latest_result.face_landmarks:
                annotated_frame = draw_landmarks_on_frame(
                    frame,
                    latest_result.face_landmarks,
                    hide_frame=hide_frame
                )
                cv2.imshow("Face Landmarks", annotated_frame)
            else:
                if hide_frame:
                    cv2.imshow("Face Landmarks", np.ones(frame.shape, dtype=np.uint8))
                else:
                    cv2.imshow("Face Landmarks", frame)
        
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

def main():
    cam = setup_camera()

    try:
        process_face_landmarks(cam, model_path)
    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()