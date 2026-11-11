"""Classifiers over windows of body-centred features, input shape (batch, 30, features)."""
import numpy as np
import torch
import torch.nn as nn

from .constants import EDGES, NUM_CLASSES, NUM_JOINTS


class SignalLSTM(nn.Module):
    """Proposed model: LayerNorm, 2-layer unidirectional LSTM (128 units), dropout, linear head.
    Unidirectional, so the hidden state can be updated frame by frame on board. 229k parameters."""

    def __init__(self, input_dim, num_classes=NUM_CLASSES, hidden=128):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=2, batch_first=True, dropout=0.2)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(hidden, num_classes))

    def forward(self, x):
        x, _ = self.lstm(self.norm(x))
        return self.head(x[:, -1])


class StackedLSTM(nn.Module):
    """Baseline from the original final-year project: LSTMs of 64, 128 and 64 units, dense 64 and 32."""

    def __init__(self, input_dim, num_classes=NUM_CLASSES):
        super().__init__()
        self.l1 = nn.LSTM(input_dim, 64, batch_first=True)
        self.l2 = nn.LSTM(64, 128, batch_first=True)
        self.l3 = nn.LSTM(128, 64, batch_first=True)
        self.head = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(),
                                  nn.Linear(32, num_classes))

    def forward(self, x):
        x, _ = self.l1(x)
        x, _ = self.l2(torch.relu(x))
        x, _ = self.l3(torch.relu(x))
        return self.head(torch.relu(x[:, -1]))


class TCN(nn.Module):
    """Temporal convolutional baseline: three dilated convolutions of 128 channels (non-causal)."""

    def __init__(self, input_dim, num_classes=NUM_CLASSES, hidden=128):
        super().__init__()
        layers = []
        for i, dilation in enumerate((1, 2, 4)):
            layers += [nn.Conv1d(input_dim if i == 0 else hidden, hidden, 5, padding=2 * dilation, dilation=dilation),
                       nn.BatchNorm1d(hidden), nn.ReLU()]
        self.net = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(hidden, num_classes))

    def forward(self, x):
        return self.head(self.net(x.transpose(1, 2)).mean(-1))


def skeleton_adjacency():
    """Symmetrically normalised adjacency of the 14-joint skeleton with self-loops."""
    A = np.eye(NUM_JOINTS, dtype=np.float32)
    for i, j in EDGES:
        A[i, j] = A[j, i] = 1
    D = np.diag(1 / np.sqrt(A.sum(1)))
    return torch.tensor(D @ A @ D)


class STGCNBlock(nn.Module):
    def __init__(self, c_in, c_out, A):
        super().__init__()
        self.register_buffer('A', A)
        self.g = nn.Conv2d(c_in, c_out, 1)
        self.t = nn.Conv2d(c_out, c_out, (9, 1), padding=(4, 0))
        self.bn1 = nn.BatchNorm2d(c_out)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.res = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):   # (B, C, T, V)
        y = torch.einsum('bctv,vw->bctw', self.g(x), self.A)
        y = self.bn2(self.t(torch.relu(self.bn1(y))))
        return torch.relu(y + self.res(x))


class STGCN(nn.Module):
    """Light spatio-temporal graph network: three blocks of 32, 64 and 64 channels."""

    def __init__(self, input_dim, num_classes=NUM_CLASSES):
        super().__init__()
        self.channels = input_dim // NUM_JOINTS
        A = skeleton_adjacency()
        self.blocks = nn.Sequential(STGCNBlock(self.channels, 32, A), STGCNBlock(32, 64, A), STGCNBlock(64, 64, A))
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(64, num_classes))

    def forward(self, x):   # (B, T, F), F laid out as [positions (V*2), velocities (V*2)]
        B, T, _ = x.shape
        x = x.view(B, T, self.channels // 2, NUM_JOINTS, 2).permute(0, 2, 4, 1, 3)
        return self.head(self.blocks(x.reshape(B, self.channels, T, NUM_JOINTS)).mean((2, 3)))


ARCHITECTURES = {'lstm': SignalLSTM, 'stacked_lstm': StackedLSTM, 'tcn': TCN, 'stgcn': STGCN}


def build_model(arch, input_dim, num_classes=NUM_CLASSES):
    return ARCHITECTURES[arch](input_dim, num_classes)
