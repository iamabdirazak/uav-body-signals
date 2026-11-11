"""Convert UAV-Human's Skeleton.zip into one compact archive for experiments.uavhuman.

UAV-Human is available for non-commercial research from https://sutdcv.github.io/uav-human-web/
(ActionRecognition > Skeleton > Skeleton.zip, 626 MB). It may not be redistributed, so this repository
does not include it.

    python scripts/prepare_uavhuman.py ~/Downloads/Skeleton.zip data/uavhuman_skeletons.npz

Each text file holds the frame count, then per frame the number of bodies and, per body, an
information line and 17 joints of 12 values each, of which the first two are pixel coordinates.
The first body in each frame is kept.
"""
import os
import sys
import zipfile

import numpy as np


def parse(text):
    t = text.split()
    i = 0
    frames = int(float(t[i]))
    i += 1
    out = np.zeros((frames, 17, 2), np.float32)
    for f in range(frames):
        bodies = int(float(t[i]))
        i += 1
        for b in range(bodies):
            i += 10
            joints = int(float(t[i]))
            i += 1
            values = np.array(t[i:i + joints * 12], float).reshape(joints, 12)[:, :2]
            i += joints * 12
            if b == 0:
                out[f, :joints] = values
    return out


def main():
    src, dst = sys.argv[1], sys.argv[2]
    clips = {}
    with zipfile.ZipFile(src) as z:
        for name in sorted(n for n in z.namelist() if n.endswith('.txt')):
            clips[os.path.basename(name)[:-4]] = parse(z.read(name).decode()).astype(np.float16)
    os.makedirs(os.path.dirname(dst) or '.', exist_ok=True)
    np.savez_compressed(dst, **clips)
    print(f'{len(clips)} clips written to {dst}')


if __name__ == '__main__':
    main()
