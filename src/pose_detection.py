
import cv2
import mediapipe as mp


class PoseDetector:

    # MediaPipe Pose landmark indexes
    LANDMARKS = {
        "left_shoulder": 11,
        "right_shoulder": 12,
        "left_elbow": 13,
        "right_elbow": 14,
        "left_wrist": 15,
        "right_wrist": 16
    }

    # Only draw the arm landmarks we need
    CONNECTIONS = [
        (11, 13),  # Left shoulder -> elbow
        (13, 15),  # Left elbow -> wrist
        (12, 14),  # Right shoulder -> elbow
        (14, 16)   # Right elbow -> wrist
    ]

    def __init__(self, model_path):

        self.mp = mp

        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=model_path
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.detector = mp.tasks.vision.PoseLandmarker.create_from_options(
            options
        )

        self.frame_timestamp = 0

    def detect(self, frame):

        # OpenCV uses BGR, MediaPipe expects RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = self.mp.Image(
            image_format=self.mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Video timestamps must increase
        self.frame_timestamp += 1

        timestamp_ms = self.frame_timestamp * 33

        results = self.detector.detect_for_video(
            mp_image,
            timestamp_ms
        )

        if not results.pose_landmarks:
            return frame, None

        landmarks = results.pose_landmarks[0]

        # Draw arm landmarks
        self._draw_landmarks(frame, landmarks)

        return frame, landmarks

    def _draw_landmarks(self, frame, landmarks):

        height, width, _ = frame.shape

        # Convert normalized coordinates to pixels
        points = {}

        for name, index in self.LANDMARKS.items():

            landmark = landmarks[index]

            visibility = landmark.visibility

            if visibility < 0.5:
                continue

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            points[index] = (x, y)

            cv2.circle(
                frame,
                (x, y),
                7,
                (0, 255, 0),
                -1
            )

            cv2.putText(
                frame,
                name,
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )

        # Draw connections between detected points
        for start, end in self.CONNECTIONS:

            if start in points and end in points:

                cv2.line(
                    frame,
                    points[start],
                    points[end],
                    (255, 0, 0),
                    3
                )

    def close(self):

        self.detector.close()
