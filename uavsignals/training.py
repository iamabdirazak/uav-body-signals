"""Training, prediction and evaluation on the UAV-GESTURE folds.

The configurations form the model comparison in the paper. Every model is trained with the
same schedule (35 epochs) and no hyperparameter tuning, once per fold and seed.
"""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .augment import augment
from .constants import WINDOW
from .features import pad_to_window, sliding_windows, window_features
from .models import Standardised, build_model

TRAIN_STRIDE = 5    # training windows every 5 frames
TEST_STRIDE = 15    # test windows every 15 frames


@dataclass(frozen=True)
class Config:
    arch: str            # key of models.ARCHITECTURES
    norm: str            # 'body' or 'image'
    velocity: bool       # append velocities to positions
    augment: bool        # viewpoint augmentation during training
    standardise: bool = False      # z-score each input feature with training-set statistics
    epochs: int = 35

    def features(self, x, visible):
        return window_features(x, visible, self.norm, self.velocity)

    @property
    def input_dim(self):
        return 14 * 2 * (2 if self.velocity else 1)


CONFIGS = {
    'stacked_lstm_image': Config('stacked_lstm', 'image', velocity=False, augment=False, standardise=True),
    'stacked_lstm_body':  Config('stacked_lstm', 'body', velocity=False, augment=False),
    'lstm_no_aug':        Config('lstm', 'body', velocity=True, augment=False),
    'lstm':               Config('lstm', 'body', velocity=True, augment=True),       # proposed
    'tcn':                Config('tcn', 'body', velocity=True, augment=True),
    'stgcn':              Config('stgcn', 'body', velocity=True, augment=True),
}


def make_model(cfg):
    """The network for a configuration, wrapped in input standardisation when the configuration uses it."""
    net = build_model(cfg.arch, cfg.input_dim)
    return Standardised(net, cfg.input_dim) if cfg.standardise else net


def train(cfg, videos, names, seed=0, epochs=None, threads=4):
    """Train one model on the given videos and return it in eval mode."""
    torch.set_num_threads(threads)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    epochs = epochs or cfg.epochs
    samples = [(w, videos[n].label) for n in names
               for w in sliding_windows(videos[n].keypoints, videos[n].visible, TRAIN_STRIDE)]
    y = torch.tensor([label for _, label in samples])
    net = make_model(cfg)
    fixed = None if cfg.augment else torch.tensor(np.stack([cfg.features(x, m) for (x, m), _ in samples]))
    if cfg.standardise:
        net.fit(fixed.reshape(-1, cfg.input_dim))
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
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
    """Window accuracy, video accuracy (mean of window probabilities) and the predicted class of each video."""
    correct_w = total_w = correct_v = 0
    predictions = {}
    for n in names:
        v = videos[n]
        P = predict(net, cfg, v.keypoints, v.visible, perturb=perturb)
        predictions[n] = int(P.mean(0).argmax())
        correct_w += (P.argmax(1) == v.label).sum()
        total_w += len(P)
        correct_v += predictions[n] == v.label
    return float(correct_w / total_w), float(correct_v / len(names)), predictions


def load_model(cfg, path):
    net = make_model(cfg)
    net.load_state_dict(torch.load(path, map_location='cpu'))
    return net.eval()
