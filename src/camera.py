
import cv2


class Camera:

    def __init__(self, camera_index=0, width=1280, height=720):

        self.camera = cv2.VideoCapture(camera_index)

        if not self.camera.isOpened():
            raise RuntimeError("Camera could not be opened.")

        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def get_frame(self):

        success, frame = self.camera.read()

        if not success:
            return None

        return frame

    def release(self):

        if self.camera is not None:
            self.camera.release()
