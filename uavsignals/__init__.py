"""Viewpoint-robust body-signal recognition for commanding UAVs."""
from .constants import CLASSES
from .gate import SafetyGate
from .models import SignalLSTM, build_model
from .recogniser import Output, Recogniser

__all__ = ['CLASSES', 'SafetyGate', 'SignalLSTM', 'build_model', 'Recogniser', 'Output']
__version__ = '1.0.0'
