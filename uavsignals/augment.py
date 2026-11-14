"""Perturbations of raw keypoints that approximate what an on-board pose estimator sees.

They act on image-space keypoints (T, 14, 2) and the visibility mask, before normalisation.
`augment` combines them for training; the robustness study applies each one alone, at test time.

  dropout         joints the detector misses (occlusion, low light, a small person)
  limb_occlusion  one arm hidden for 10 to 30 consecutive frames
  jitter          localisation noise, in torso lengths (blur, distance, low contrast)
  elevation       camera pitch: first-order vertical foreshortening by cos(theta) about the neck
  rotate          camera roll or airframe tilt
  low_fps         pose estimator running below 25 Hz, held to the 25 Hz timeline
"""
import numpy as np

from .constants import FPS, L_ARM, NECK, R_ARM
from .features import torso_length


def dropout(x, visible, p, rng):
    return x, visible & (rng.random(visible.shape) > p)


def limb_occlusion(x, visible, p, rng):
    """With probability p, hide the elbow and wrist of one arm for a random 10 to 30 frame span."""
    visible = visible.copy()
    if rng.random() < p:
        arm = R_ARM if rng.random() < .5 else L_ARM
        T = len(visible)
        span = rng.integers(10, 31)
        start = rng.integers(0, max(1, T - span))
        visible[start:start + span, list(arm[1:])] = False
    return x, visible


def jitter(x, visible, sigma, rng):
    noise = rng.normal(0, sigma * (torso_length(x) + 1e-6), x.shape).astype(np.float32)
    return x + noise, visible


def elevation(x, visible, degrees):
    x = x.copy()
    neck_y = x[:, NECK:NECK + 1, 1]
    x[..., 1] = neck_y + (x[..., 1] - neck_y) * np.cos(np.deg2rad(degrees))
    return x, visible


def rotate(x, visible, degrees):
    a = np.deg2rad(degrees)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], np.float32)
    neck = x[:, NECK:NECK + 1]
    return (x - neck) @ R.T + neck, visible


def low_fps(x, visible, fps, base=FPS):
    step = base / fps
    idx = np.floor(np.floor(np.arange(len(x)) / step) * step).astype(int)
    return x[idx], visible[idx]


def augment(x, visible, rng):
    """Training-time viewpoint augmentation."""
    x, visible = rotate(x, visible, rng.uniform(-10, 10))
    x, visible = elevation(x, visible, rng.uniform(0, 60))
    x = x * rng.uniform(0.8, 1.2, (1, 1, 2)).astype(np.float32)
    x, visible = jitter(x, visible, rng.uniform(0, 0.08), rng)
    x, visible = dropout(x, visible, rng.uniform(0, 0.3), rng)
    x, visible = limb_occlusion(x, visible, 0.3, rng)
    if rng.random() < 0.3:
        x, visible = low_fps(x, visible, rng.choice([5, 8.33, 12.5]))
    return x, visible


def perturbation(kind, value):
    """Return f(x, visible, rng) applying one test-time condition at a given strength."""
    table = {
        'dropout': lambda x, m, r: dropout(x, m, value, r),
        'arm_occlusion': lambda x, m, r: limb_occlusion(x, m, value, r),
        'jitter': lambda x, m, r: jitter(x, m, value, r),
        'elevation': lambda x, m, r: elevation(x, m, value),
        'rotation': lambda x, m, r: rotate(x, m, value),
        'fps': lambda x, m, r: low_fps(x, m, value),
    }
    return table[kind]
