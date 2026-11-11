"""Summary statistics used across the experiments.

Each configuration is trained with several seeds. For one seed, accuracy is pooled over the test videos
of all three folds (93 videos), so every video counts once. The spread across seeds gives the 95%
confidence interval of the mean. Models are compared on the same videos with a paired test.
"""
import numpy as np
from scipy import stats


def mean_ci(values, level=0.95):
    """Mean and half-width of the t-based confidence interval across seeds."""
    values = np.asarray(values, float)
    if len(values) < 2:
        return float(values.mean()), 0.0
    half = stats.t.ppf(0.5 + level / 2, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    return float(values.mean()), float(half)


def paired_test(correct_a, correct_b, permutations=20000, seed=0):
    """Compare two models on the same videos.

    correct_a, correct_b: (videos,) fraction of seeds that classified each video correctly.
    Returns the mean difference (a - b), its 95% bootstrap interval, and a two-sided p-value from a
    sign-flip permutation test on the per-video differences.
    """
    d = np.asarray(correct_a, float) - np.asarray(correct_b, float)
    rng = np.random.default_rng(seed)
    observed = abs(d.mean())
    flips = rng.choice([-1.0, 1.0], size=(permutations, len(d)))
    p = (np.sum(np.abs((flips * d).mean(1)) >= observed - 1e-12) + 1) / (permutations + 1)
    boots = rng.choice(d, size=(permutations, len(d))).mean(1)
    return dict(difference=float(d.mean()), ci=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                p=float(p))
