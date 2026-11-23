"""Metrics with participant-clustered uncertainty.

Each participant contributes several pairs, so confidence intervals resample participants
(with all of their pairs) rather than individual pairs.
"""
import random
from collections import defaultdict


def cluster_bootstrap_ci(items, statistic, cluster=lambda x: x["participant_id"],
                         n_boot=10_000, seed=0, level=0.95):
    groups = defaultdict(list)
    for item in items:
        groups[cluster(item)].append(item)
    groups = list(groups.values())
    rng = random.Random(seed)
    values = sorted(statistic([x for g in rng.choices(groups, k=len(groups)) for x in g])
                    for _ in range(n_boot))
    lo = values[int((1 - level) / 2 * n_boot)]
    hi = values[int((1 + level) / 2 * n_boot) - 1]
    return lo, hi


def correct(score, y):
    """1 for a correct prediction, 0.5 for an exact tie (expected random-choice credit)."""
    if score == 0:
        return 0.5
    return float((score > 0) == bool(y))


def accuracy(rows):
    """rows: dicts with 'score' (memory-oriented) and 'y' (1 = memory preferred)."""
    return sum(correct(r["score"], r["y"]) for r in rows) / len(rows)


def majority_accuracy(rows):
    """Accuracy of always predicting the more frequent outcome within `rows`."""
    k = sum(r["y"] for r in rows)
    return max(k, len(rows) - k) / len(rows)


def predicted_memory_rate(rows):
    return sum(0.5 if r["score"] == 0 else float(r["score"] > 0) for r in rows) / len(rows)


def coverage_accuracy(rows, coverage):
    """Accuracy on the `coverage` fraction of rows with the largest |score| (Table 5)."""
    ranked = sorted(rows, key=lambda r: -abs(r["score"]))
    keep = ranked[:max(1, round(coverage * len(rows)))]
    return accuracy(keep)
