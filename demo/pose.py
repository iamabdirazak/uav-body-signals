"""MediaPipe pose estimation, mapped to the 14 COCO body joints the model expects."""
import os
import urllib.request

import numpy as np

MODEL_URL = ('https://storage.googleapis.com/mediapipe-models/pose_landmarker/'
             'pose_landmarker_lite/float16/latest/pose_landmarker_lite.task')

# our joint index -> MediaPipe landmark index (the neck is the midpoint of the shoulders)
MEDIAPIPE_TO_COCO = {0: 0, 2: 12, 3: 14, 4: 16, 5: 11, 6: 13, 7: 15, 8: 24, 9: 26, 10: 28, 11: 23, 12: 25, 13: 27}


def to_coco14(landmarks, width, height, min_visibility=0.5):
    """MediaPipe landmarks (normalised) -> (14, 2) pixel keypoints and (14,) visibility."""
    keypoints = np.zeros((14, 2), np.float32)
    visible = np.zeros(14, bool)
    for j, i in MEDIAPIPE_TO_COCO.items():
        lm = landmarks[i]
        keypoints[j] = (lm.x * width, lm.y * height)
        visible[j] = getattr(lm, 'visibility', 1.0) >= min_visibility and 0 <= lm.x <= 1 and 0 <= lm.y <= 1
    keypoints[1] = (keypoints[2] + keypoints[5]) / 2
    visible[1] = visible[2] and visible[5]
    return keypoints, visible


class PoseEstimator:
    def __init__(self, model_path):
        from mediapipe.tasks.python import BaseOptions, vision
        if not os.path.exists(model_path):
            print('Downloading the MediaPipe pose model (about 6 MB)...')
            urllib.request.urlretrieve(MODEL_URL, model_path)
        self.landmarker = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO, num_poses=1))

    def __call__(self, frame_bgr, t):
        """Returns (keypoints, visible) for the most prominent person, or (None, None)."""
        import cv2
        import mediapipe as mp
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        result = self.landmarker.detect_for_video(image, int(t * 1000))
        if not result.pose_landmarks:
            return None, None
        h, w = frame_bgr.shape[:2]
        return to_coco14(result.pose_landmarks[0], w, h)
