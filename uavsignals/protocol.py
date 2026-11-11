"""Actor-disjoint evaluation folds.

Folds 2 and 3 are the dataset's official splits 2 and 3. Official split 1 lists actor S12 in
both its training list (7 videos) and its test list (6 videos), so all S12 videos are moved to
training. No actor appears in both training and test in any fold.

The dataset paper reports 8 participants recorded on 2 days under 11 actor IDs, so the same
physical person may still appear under two IDs; the released data cannot resolve this.
"""

TEST_ACTORS = {
    1: {'S13', 'S14', 'S15'},
    2: {'S1', 'S3', 'S4'},
    3: {'S5', 'S6', 'S9'},
}
FOLDS = tuple(TEST_ACTORS)


def fold(videos, k):
    """Return sorted (train names, test names) for fold k."""
    test = sorted(n for n, v in videos.items() if v.actor in TEST_ACTORS[k])
    train = sorted(n for n, v in videos.items() if v.actor not in TEST_ACTORS[k])
    return train, test
