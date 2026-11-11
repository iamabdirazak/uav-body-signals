"""Loading the UAV-GESTURE keypoints.

Each video ships as a MATLAB file holding OpenPose COCO-18 joints with shape (2, 18, T).
A missing joint is stored as (0, 0).
"""
import glob
import os
from dataclasses import dataclass

import numpy as np
import scipy.io as sio

from .constants import CLASSES, NUM_JOINTS

DEFAULT_ROOT = os.environ.get('UAVG_ROOT', os.path.join('data', 'UAV-GESTURE', 'UAVGESTURE'))


@dataclass
class Video:
    keypoints: np.ndarray   # (T, 14, 2) image coordinates in pixels
    visible: np.ndarray     # (T, 14) bool, False where the pose estimator found no joint
    label: int              # index into CLASSES (-1 for unlabelled streams)
    actor: str              # actor ID, e.g. 'S1'


def load_dataset(root=DEFAULT_ROOT):
    """Return {video name: Video} for all 119 videos."""
    pattern = os.path.join(root, 'joint_positions', '{}', '*', '*.mat')
    videos = {}
    for label, cls in enumerate(CLASSES):
        for path in sorted(glob.glob(pattern.format(cls))):
            name = os.path.basename(os.path.dirname(path))
            raw = sio.loadmat(path)['pos_img'].astype(np.float32)          # (2, 18, T)
            keypoints = np.transpose(raw, (2, 1, 0))[:, :NUM_JOINTS, :]     # (T, 14, 2)
            visible = ~(keypoints == 0).all(-1)
            videos[name] = Video(keypoints, visible, label, name.split('_')[0])
    if not videos:
        raise FileNotFoundError(
            f'No keypoint files found under {root!r}. Run scripts/get_data.sh '
            'or set UAVG_ROOT to the UAVGESTURE folder.')
    return videos
