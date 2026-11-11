"""Loading UAV-Human skeletons for cross-dataset evaluation.

UAV-Human (Li et al., CVPR 2021) records 155 actions from a UAV hovering 2 to 8 m high, with 2D poses
of 17 COCO joints estimated automatically. Its UAV-control gestures include the 13 UAV-GESTURE signals
under other names. The data cannot be redistributed, so it is read from a local archive prepared by
scripts/prepare_uavhuman.py.
"""
import re

import numpy as np

from .constants import CLASSES

# UAV-Human action -> UAV-GESTURE signal
SIGNAL_OF_ACTION = {
    'A099': 'hover', 'A100': 'land', 'A101': 'landing_direction', 'A102': 'move_ahead', 'A104': 'move_to_left',
    'A105': 'move_to_right', 'A106': 'move_upward', 'A107': 'move_downward', 'A109': 'slow_down',
    'A114': 'wave_off', 'A115': 'all_clear', 'A116': 'not_clear', 'A117': 'have_command',
}
UAV_CONTROL = {f'A{i:03d}' for i in range(99, 121)}     # every UAV-control gesture in UAV-Human

# our 14 joints from COCO-17 (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles, left before right)
COCO17_TO_14 = [0, None, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15]


def to_14(coco):
    """(T, 17, 2) pixels with zeros for missing joints -> (T, 14, 2) keypoints and (T, 14) visibility."""
    coco = coco.astype(np.float32)
    seen = ~(coco == 0).all(-1)
    x = np.zeros((len(coco), 14, 2), np.float32)
    visible = np.zeros((len(coco), 14), bool)
    for j, c in enumerate(COCO17_TO_14):
        if c is None:                                      # neck: midpoint of the shoulders
            x[:, j] = (coco[:, 5] + coco[:, 6]) / 2
            visible[:, j] = seen[:, 5] & seen[:, 6]
        else:
            x[:, j], visible[:, j] = coco[:, c], seen[:, c]
    return x, visible


def load_uavhuman(path):
    """{clip name: dict(keypoints, visible, action, person, signal)} from the prepared archive."""
    clips = {}
    with np.load(path) as data:
        for name in data.files:
            action = re.search(r'A\d{3}', name).group()
            x, visible = to_14(data[name])
            if len(x) == 0 or visible[:, [1, 8, 11]].all(1).sum() < 15:
                continue                                    # too short, or the torso is hardly ever found
            signal = SIGNAL_OF_ACTION.get(action)
            clips[name] = dict(keypoints=x, visible=visible, action=action, person=name[:4],
                               signal=CLASSES.index(signal) if signal else None)
    return clips
