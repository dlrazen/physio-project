
import cv2

from src.camera import Camera
from src.pose_detection import PoseDetector
from src.angle_calculator import AngleCalculator


def main():

    camera = None
    pose_detector = None

    try:

        camera = Camera(
            camera_index=0,
            width=1280,
            height=720
        )

        pose_detector = PoseDetector(
            model_path="models/pose_landmarker_lite.task"
        )

        angle_calculator = AngleCalculator(
            min_visibility=0.5
        )

        # Select the arm to measure
        selected_side = "right"

        print("Pose detection started.")
        print("Measuring:", selected_side, "arm")
        print("Press Q to exit.")

        while True:

            frame = camera.get_frame()

            if frame is None:
                print("Failed to read camera frame.")
                break

            processed_frame, landmarks = pose_detector.detect(frame)

            angle = None

            if landmarks is not None:

                angle = angle_calculator.calculate_from_landmarks(
                    landmarks=landmarks,
                    frame_shape=frame.shape,
                    side=selected_side
                )

            # Display angle
            if angle is not None:

                cv2.putText(
                    processed_frame,
                    f"Elbow Flexion: {angle:.2f} deg",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

            else:

                cv2.putText(
                    processed_frame,
                    "Angle unavailable",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )

            cv2.imshow(
                "Physio Project - Elbow Angle Measurement",
                processed_frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception as error:

        print(f"Error: {error}")

    finally:

        if pose_detector is not None:
            pose_detector.close()

        if camera is not None:
            camera.release()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
