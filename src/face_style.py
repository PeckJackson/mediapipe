import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np

from utils.model_loader import get_face_model_path
from utils.camera_setup import setup_camera

MODEL_STYLE = "face_stylizer_oil_painting"

style_path = get_face_model_path(MODEL_STYLE)


#### Styler

# def setup_styler(style_path):
#     style_transfer = mp.solutions.style_transfer
#     styler = style_transfer.StyleTransfer(
#         model_path=style_path,
#         num_slots=1
#     )
#     return styler


#### Open cam and apply style


def setup_styler_options(style_path):
    base_options = python.BaseOptions(model_asset_path=style_path)
    options = vision.FaceStylizerOptions(base_options=base_options)
    return options

def process_and_display_frame(cam, style_path):

    options = setup_styler_options(style_path)

    with vision.FaceStylizer.create_from_options(options) as stylizer:

        while True:
            ret, frame = cam.read()
            if not ret:
                raise IOError("Could not read camera data")

            # cv2.imshow("test",frame)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(frame_rgb)
            )

            # mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            stylized_image = landmarker.detect_async(mp_image, frame_timestamp_ms)

            if stylized_image is not None:
                # Convert MediaPipe Image to numpy array and then to BGR
                output_frame = cv2.cvtColor(
                    np.array(stylized_image.numpy_view()), 
                    cv2.COLOR_RGB2BGR
                )
                cv2.imshow("Stylized Face", output_frame)
            else:
                # If no face detected, show original frame
                cv2.imshow("Stylized Face", frame)
        
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

def main():
    cam = setup_camera()

    try:
        process_and_display_frame(cam, style_path)
    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()