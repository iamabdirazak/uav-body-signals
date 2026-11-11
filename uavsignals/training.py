"""Training, prediction and evaluation on the UAV-GESTURE folds.

The configurations reproduce the ablation in Table I of the paper. Each is trained once per fold
with seed 0 and no hyperparameter tuning.
"""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .augment import augment
from .constants import NUM_CLASSES, WINDOW
from .features import pad_to_window, sliding_windows, window_features
from .models import build_model

TRAIN_STRIDE = 5    # training windows every 5 frames
TEST_STRIDE = 15    # test windows every 15 frames


@dataclass(frozen=True)
class Config:
    arch: str            # key of models.ARCHITECTURES
    norm: str            # 'body' or 'image'
    velocity: bool       # append velocities to positions
    augment: bool        # viewpoint augmentation during training
    epochs: tuple = (35, 35, 35)   # epochs for folds 1, 2, 3, as run for the paper

    def features(self, x, visible):
        return window_features(x, visible, self.norm, self.velocity)

    @property
    def input_dim(self):
        return 14 * 2 * (2 if self.velocity else 1)


CONFIGS = {
    'stacked_lstm_image': Config('stacked_lstm', 'image', velocity=False, augment=False, epochs=(40, 40, 40)),
    'stacked_lstm_body':  Config('stacked_lstm', 'body', velocity=False, augment=False, epochs=(40, 40, 40)),
    'lstm_no_aug':        Config('lstm', 'body', velocity=True, augment=False, epochs=(40, 35, 35)),
    'lstm':               Config('lstm', 'body', velocity=True, augment=True),       # proposed
    'tcn':                Config('tcn', 'body', velocity=True, augment=True),
    'stgcn':              Config('stgcn', 'body', velocity=True, augment=True),
}


def train(cfg, videos, names, epochs, seed=0, threads=4):
    """Train one model on the given videos and return it in eval mode."""
    torch.set_num_threads(threads)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    samples = [(w, videos[n].label) for n in names
               for w in sliding_windows(videos[n].keypoints, videos[n].visible, TRAIN_STRIDE)]
    y = torch.tensor([label for _, label in samples])
    net = build_model(cfg.arch, cfg.input_dim)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    fixed = None if cfg.augment else torch.tensor(np.stack([cfg.features(x, m) for (x, m), _ in samples]))
    for _ in range(epochs):
        # augmentation draws a fresh perturbation of every window each epoch
        X = fixed if fixed is not None else torch.tensor(
            np.stack([cfg.features(*augment(x, m, rng)) for (x, m), _ in samples]))
        net.train()
        order = torch.randperm(len(X))
        for i in range(0, len(X), 64):
            batch = order[i:i + 64]
            loss = nn.functional.cross_entropy(net(X[batch]), y[batch], label_smoothing=0.05)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    return net.eval()


@torch.no_grad()
def predict(net, cfg, x, visible, stride=TEST_STRIDE, perturb=None, seed=0):
    """Window probabilities (num_windows, 13) for one keypoint sequence."""
    if perturb is not None:
        x, visible = perturb(x, visible, np.random.default_rng(seed))
    x, visible = pad_to_window(x, visible)
    X = np.stack([cfg.features(x[i:i + WINDOW], visible[i:i + WINDOW])
                  for i in range(0, len(x) - WINDOW + 1, stride)])
    return torch.softmax(net(torch.tensor(X)), -1).numpy()


def evaluate(net, cfg, videos, names, perturb=None):
    """Window accuracy, video accuracy (mean of window probabilities) and the video confusion matrix."""
    correct_w = total_w = correct_v = 0
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), int)
    for n in names:
        v = videos[n]
        P = predict(net, cfg, v.keypoints, v.visible, perturb=perturb)
        pred_video = P.mean(0).argmax()
        correct_w += (P.argmax(1) == v.label).sum()
        total_w += len(P)
        correct_v += pred_video == v.label
        confusion[v.label, pred_video] += 1
    return correct_w / total_w, correct_v / len(names), confusion


def load_model(cfg, path):
    net = build_model(cfg.arch, cfg.input_dim)
    net.load_state_dict(torch.load(path, map_location='cpu'))
    return net.eval()
