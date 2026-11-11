"""End-to-end timing on this computer: camera frame, MediaPipe pose estimation, then the recogniser.

Stand in front of the camera for the duration. Nothing is drawn; the result is printed and saved.

    python demo/benchmark.py                 # 30 s from the first camera that works
    python demo/benchmark.py --seconds 60 --camera 1
"""
import argparse
import collections
import json
import os
import platform
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import torch  # noqa: E402

from uavsignals import Recogniser  # noqa: E402

from camera import find_camera  # noqa: E402
from pose import PoseEstimator  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seconds', type=float, default=30)
    ap.add_argument('--camera', type=int, default=-1)
    ap.add_argument('--pose-model', default=os.path.join(HERE, 'pose_landmarker_lite.task'))
    args = ap.parse_args()
    torch.set_num_threads(1)

    pose = PoseEstimator(args.pose_model)
    recogniser = Recogniser.from_pretrained()
    cap = find_camera(args.camera)
    if cap is None:
        sys.exit('No camera gave a picture.')

    # The classifier is timed on every fifth frame, on the last 30 frames of keypoints, whether or not the
    # whole body is in view, so the measurement does not depend on how the camera frames the person.
    pose_ms, decision_ms, frame_ms, detected = [], [], [], 0
    recent = collections.deque(maxlen=30)
    start = time.time()
    while time.time() - start < args.seconds:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        keypoints, visible = pose(frame, time.time() - start)
        t1 = time.perf_counter()
        pose_ms.append((t1 - t0) * 1000)
        if keypoints is not None:
            detected += 1
            recent.append((keypoints, visible))
            if len(recent) == 30 and detected % 5 == 0:
                x = np.stack([k for k, _ in recent])
                m = np.stack([v for _, v in recent])
                m[:, 1] = True                               # keep the neck anchor so features can be computed
                t2 = time.perf_counter()
                recogniser.classify(x, m)
                decision_ms.append((time.perf_counter() - t2) * 1000)
        frame_ms.append((time.perf_counter() - t0) * 1000)
    cap.release()

    result = dict(
        machine=f'{platform.system()} {platform.machine()}', processor=platform.processor(),
        frames=len(pose_ms), person_detected=detected / max(1, len(pose_ms)),
        pose_ms_median=float(np.median(pose_ms)),
        recogniser_ms_per_decision_median=float(np.median(decision_ms)) if decision_ms else None,
        decisions=len(decision_ms),
        pose_fps=1000 / float(np.median(pose_ms)))
    print(json.dumps(result, indent=1))
    os.makedirs(os.path.join(ROOT, 'runs', 'results'), exist_ok=True)
    with open(os.path.join(ROOT, 'runs', 'results', 'timing.json'), 'w') as f:
        json.dump(result, f, indent=1)


if __name__ == '__main__':
    main()
