"""Fast checks that need no dataset: shapes, the released weights, and the safety gate's rules."""
import numpy as np
import torch

from uavsignals import CLASSES, Recogniser, SafetyGate
from uavsignals.constants import FEATURE_DIM, HAVE_COMMAND, NUM_JOINTS, WAVE_OFF, WINDOW
from uavsignals.features import window_features
from uavsignals.models import ARCHITECTURES


def standing_person(T=WINDOW, rng=None):
    rng = rng or np.random.default_rng(0)
    base = np.array([[960, 300], [960, 360], [900, 360], [880, 440], [870, 520], [1020, 360], [1040, 440],
                     [1050, 520], [930, 560], [930, 680], [930, 800], [990, 560], [990, 680], [990, 800]], np.float32)
    return base[None] + rng.normal(0, 2, (T, NUM_JOINTS, 2)).astype(np.float32)


def onehot(c, p=0.95):
    v = np.full(len(CLASSES), (1 - p) / (len(CLASSES) - 1))
    v[c] = p
    return v


def test_features_are_position_and_scale_invariant():
    x = standing_person()
    m = np.ones(x.shape[:2], bool)
    f = window_features(x, m)
    assert f.shape == (WINDOW, FEATURE_DIM)
    moved = window_features(x * 0.5 + 300, m)
    assert np.allclose(f, moved, atol=1e-4)


def test_missing_joints_are_filled():
    x = standing_person()
    m = np.ones(x.shape[:2], bool)
    m[5:15, 4] = False
    assert np.isfinite(window_features(x, m)).all()


def test_all_architectures_run():
    x = torch.randn(2, WINDOW, FEATURE_DIM)
    for arch in ARCHITECTURES.values():
        assert arch(FEATURE_DIM).eval()(x).shape == (2, len(CLASSES))


def test_released_weights_stream():
    rec = Recogniser.from_pretrained()
    x = standing_person(T=60)
    outputs = [rec.push(t / 25, x[t], np.ones(NUM_JOINTS, bool)) for t in range(60)]
    outputs = [o for o in outputs if o is not None]
    assert outputs, 'the recogniser should classify once 1.2 s of keypoints is buffered'
    assert np.isclose(outputs[-1].probs.sum(), 1, atol=1e-5)


def test_gate_needs_attention_before_a_command():
    gate = SafetyGate(k=3, n=4, tau=0.8)
    hover = CLASSES.index('hover')
    assert all(gate.step(onehot(hover)) is None for _ in range(10))      # not armed: nothing happens
    events = [gate.step(onehot(HAVE_COMMAND)) for _ in range(4)]
    assert ('armed', HAVE_COMMAND) in events
    events = [gate.step(onehot(hover)) for _ in range(4)]
    assert ('execute', hover) in events and not gate.armed                # one command, then disarmed


def test_wave_off_is_never_gated():
    gate = SafetyGate(k=3, n=4, tau=0.8)
    events = [gate.step(onehot(WAVE_OFF)) for _ in range(4)]
    assert ('execute', WAVE_OFF) in events


def test_low_confidence_is_ignored():
    gate = SafetyGate(k=3, n=4, tau=0.8)
    assert all(gate.step(onehot(WAVE_OFF, p=0.5)) is None for _ in range(8))
