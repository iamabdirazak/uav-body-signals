"""Safety gate: decides when a stream of window predictions becomes a command.

Called once per decision step (every 0.2 s) with the latest class probabilities.

  naive  act whenever the top class changes (what a plain classifier does)
  vote   act when the majority class holds k of the last n windows with mean confidence >= tau
  gate   vote, plus an attention signal: an accepted `have command` arms the gate, the next
         accepted signal is executed and the gate disarms. `wave off` (abort) bypasses the gate.

Recognising `have command` is not itself an action.
"""
from collections import deque

import numpy as np

from .constants import HAVE_COMMAND, WAVE_OFF


class SafetyGate:
    def __init__(self, mode='gate', k=6, n=8, tau=0.7):
        if mode not in ('naive', 'vote', 'gate'):
            raise ValueError(f'unknown mode {mode!r}')
        self.mode, self.k, self.n, self.tau = mode, k, n, tau
        self.reset()

    def reset(self):
        self.history = deque(maxlen=self.n)
        self.armed = False
        self.last = None

    def step(self, probs):
        """Feed one probability vector. Returns ('execute', class), ('armed', HAVE_COMMAND) or None."""
        top = int(np.argmax(probs))
        self.history.append((top, float(np.max(probs))))
        if self.mode == 'naive':
            if top != self.last:
                self.last = top
                return 'execute', top
            return None

        classes = [c for c, _ in self.history]
        majority = max(set(classes), key=classes.count)
        confidence = np.mean([p for c, p in self.history if c == majority])
        if classes.count(majority) < self.k or confidence < self.tau:
            return None

        if self.mode == 'vote':
            if majority != self.last:
                self.last = majority
                return 'execute', majority
            return None

        if majority == WAVE_OFF and self.last != WAVE_OFF:          # abort is never gated
            self.last, self.armed = WAVE_OFF, False
            return 'execute', WAVE_OFF
        if majority == HAVE_COMMAND:
            newly_armed = not self.armed
            self.last, self.armed = HAVE_COMMAND, True
            return ('armed', HAVE_COMMAND) if newly_armed else None
        if self.armed and majority != self.last:
            self.last, self.armed = majority, False
            return 'execute', majority
        return None


def run(probs_sequence, **gate_args):
    """Run a gate over a whole sequence; return [(step, class)] of executed commands."""
    gate = SafetyGate(**gate_args)
    executed = []
    for t, p in enumerate(probs_sequence):
        event = gate.step(p)
        if event and event[0] == 'execute':
            executed.append((t, event[1]))
    return executed
