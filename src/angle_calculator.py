
import math
import numpy as np


class AngleCalculator:

    LANDMARKS = {
        "left_shoulder": 11,
        "right_shoulder": 12,
        "left_elbow": 13,
        "right_elbow": 14,
        "left_wrist": 15,
        "right_wrist": 16
    }

    def __init__(self, min_visibility=0.5):
        self.min_visibility = min_visibility

    def calculate_angle(self, shoulder, elbow, wrist):
        """
        Calculate elbow flexion angle using three 2D points.

        Points are given as (x, y) pixel coordinates.
        """

        a = np.array(shoulder, dtype=float) - np.array(elbow, dtype=float)
        b = np.array(wrist, dtype=float) - np.array(elbow, dtype=float)

        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        # Prevent division by zero
        if norm_a == 0 or norm_b == 0:
            return None

        cosine = np.dot(a, b) / (norm_a * norm_b)

        # Prevent numerical errors
        cosine = np.clip(cosine, -1.0, 1.0)

        internal_angle = math.degrees(math.acos(cosine))

        flexion_angle = 180.0 - internal_angle

        return round(flexion_angle, 2)

    def calculate_from_landmarks(
        self,
        landmarks,
        frame_shape,
        side="right"
    ):
        """
        Calculate elbow flexion angle directly from MediaPipe landmarks.

        side: 'left' or 'right'
        frame_shape: frame.shape from OpenCV
        """

        if side not in ("left", "right"):
            raise ValueError("Side must be 'left' or 'right'.")

        height, width = frame_shape[:2]

        shoulder = landmarks[
            self.LANDMARKS[f"{side}_shoulder"]
        ]

        elbow = landmarks[
            self.LANDMARKS[f"{side}_elbow"]
        ]

        wrist = landmarks[
            self.LANDMARKS[f"{side}_wrist"]
        ]

        # Check landmark visibility
        for point in (shoulder, elbow, wrist):

            if point.visibility < self.min_visibility:
                return None

        # Convert normalized coordinates to pixel coordinates
        shoulder_xy = (
            shoulder.x * width,
            shoulder.y * height
        )

        elbow_xy = (
            elbow.x * width,
            elbow.y * height
        )

        wrist_xy = (
            wrist.x * width,
            wrist.y * height
        )

        return self.calculate_angle(
            shoulder_xy,
            elbow_xy,
            wrist_xy
        )
