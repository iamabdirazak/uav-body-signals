"""Streaming recogniser: the deployable entry point for any pose estimator and any frame rate.

Push one frame of 14 keypoints at a time with its timestamp. Every 0.2 s the recogniser resamples
the last 1.2 s onto the 25 Hz grid the model was trained on, classifies it, and passes the result
through the safety gate.

    rec = Recogniser.from_pretrained()
    for t, keypoints, visible in my_pose_stream():
        out = rec.push(t, keypoints, visible)
        if out and out.command:
            drone.do(out.command)
"""
import os
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from .constants import CLASSES, FEATURE_DIM, FPS, L_HIP, NECK, R_HIP, WINDOW
from .features import window_features
from .gate import SafetyGate
from .models import SignalLSTM

DEFAULT_WEIGHTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'weights', 'uavsignals_lstm.pt')
STEP_SECONDS = 0.2


@dataclass
class Output:
    probs: np.ndarray       # (13,) class probabilities for the latest window
    signal: str             # most likely signal
    armed: bool             # gate state after this step
    command: str = None     # set when the gate executes a command


class Recogniser:
    def __init__(self, net, gate=None):
        self.net = net.eval()
        self.gate = gate or SafetyGate()
        self.buffer = deque(maxlen=200)
        self.next_time = None

    @classmethod
    def from_pretrained(cls, path=DEFAULT_WEIGHTS, **gate_args):
        """Load the released model (trained on all UAV-GESTURE actors)."""
        net = SignalLSTM(FEATURE_DIM)
        net.load_state_dict(torch.load(path, map_location='cpu'))
        return cls(net, SafetyGate(**gate_args) if gate_args else None)

    def reset(self):
        self.buffer.clear()
        self.next_time = None
        self.gate.reset()

    @torch.no_grad()
    def classify(self, keypoints, visible):
        """Probabilities for one window of shape (30, 14, 2) at 25 Hz."""
        x = torch.tensor(window_features(keypoints.astype(np.float32), visible))[None]
        return torch.softmax(self.net(x), -1)[0].numpy()

    def push(self, t, keypoints, visible):
        """Add a frame (time in seconds, (14, 2) pixel keypoints, (14,) bool). Returns Output or None."""
        self.buffer.append((t, np.asarray(keypoints, np.float32), np.asarray(visible, bool)))
        if self.next_time is None:
            self.next_time = t + WINDOW / FPS
        if t < self.next_time or len(self.buffer) < 2:
            return None
        self.next_time = t + STEP_SECONDS

        times = np.array([b[0] for b in self.buffer])
        if times[-1] - times[0] < (WINDOW - 1) / FPS * 0.9:
            return None
        # sample-and-hold onto the 25 Hz grid ending now
        grid = t - (WINDOW - 1 - np.arange(WINDOW)) / FPS
        idx = np.clip(np.searchsorted(times, grid, side='right') - 1, 0, len(times) - 1)
        x = np.stack([self.buffer[i][1] for i in idx])
        m = np.stack([self.buffer[i][2] for i in idx])
        if m[:, [NECK, R_HIP, L_HIP]].all(axis=1).sum() < WINDOW // 2:
            return None   # torso not visible often enough to normalise

        probs = self.classify(x, m)
        event = self.gate.step(probs)
        command = CLASSES[event[1]] if event and event[0] == 'execute' else None
        return Output(probs, CLASSES[int(probs.argmax())], self.gate.armed, command)
