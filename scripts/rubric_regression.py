"""Memory-aware rubric reward model: nested cross-validated L2 logistic regression (Sections 4.2 and 5).

Input: a CSV with a `pair_id` column and one column per rubric feature (Table 6), each signed
so that larger values favour the memory-conditioned response. features/feature_spec.json
lists the twelve features and their block (memory-related or response-related). Pairs
without a complete feature row are skipped.

Evaluation, as in the paper:
  * outer 5-fold cross-validation grouped by participant (no test participant in training);
  * inside each outer training set, 4-fold participant-grouped CV picks
    C in {0.003, 0.01, 0.03, 0.1, 0.3} by accuracy (ties go to the smaller C);
  * StandardScaler is fitted on training participants only;
  * the whole outer loop is repeated with 10 random participant-to-fold assignments and each
    pair's test predictions are averaged before computing accuracy.

  python scripts/rubric_regression.py --features features/rubric_features.csv            # Ours
  python scripts/rubric_regression.py --features ... --block memory                      # Memory only
  python scripts/rubric_regression.py --features ... --block response                    # Response only
  python scripts/rubric_regression.py --features ... --describe                          # Table 4/6 coefficients

Writes predictions/rubric_regression[_<block>].jsonl with p_memory and score = p_memory - 0.5.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from memrm.data import DATA, REPO, label, load_pairs  # noqa: E402

C_GRID = (0.003, 0.01, 0.03, 0.1, 0.3)


def grouped_folds(groups, k, rng):
    """Split row indices into k folds so that each participant lies in exactly one fold."""
    participants = rng.permutation(np.unique(groups))
    return [np.isin(groups, chunk) for chunk in np.array_split(participants, k)]


def model(c):
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, penalty="l2", max_iter=10_000))


def choose_c(x, y, groups, rng, inner_k):
    scores = []
    for c in C_GRID:
        accs = []
        for test in grouped_folds(groups, inner_k, rng):
            m = model(c).fit(x[~test], y[~test])
            accs.append(np.mean(m.predict(x[test]) == y[test]))
        scores.append(np.mean(accs))
    return C_GRID[int(np.argmax(scores))]


def nested_cv(x, y, groups, repeats, outer_k, inner_k, seed):
    rng = np.random.default_rng(seed)
    total, chosen = np.zeros(len(y)), []
    for _ in range(repeats):
        for test in grouped_folds(groups, outer_k, rng):
            c = choose_c(x[~test], y[~test], groups[~test], rng, inner_k)
            chosen.append(c)
            total[test] += model(c).fit(x[~test], y[~test]).predict_proba(x[test])[:, 1]
    return total / repeats, chosen


def load_features(path, columns):
    with open(path, newline="") as f:
        rows = {r["pair_id"]: r for r in csv.DictReader(f)}
    missing = [c for c in columns if c not in next(iter(rows.values()))]
    if missing:
        raise SystemExit(f"feature file lacks columns: {missing}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--spec", type=Path, default=REPO / "features" / "feature_spec.json")
    parser.add_argument("--block", choices=["all", "memory", "response"], default="all")
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--describe", action="store_true",
                        help="fit once on all pairs (C=0.1) and print standardised coefficients")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text())["features"]
    columns = [f["name"] for f in spec if args.block == "all" or f["block"] == args.block]
    feats = load_features(args.features, columns)
    pairs = load_pairs(args.data)
    complete = lambda pid: pid in feats and all(feats[pid][c] != "" for c in columns)  # noqa: E731
    absent = [p["pair_id"] for p in pairs if not complete(p["pair_id"])]
    if absent:
        print(f"skipping {len(absent)} pairs without features (e.g. {absent[0]})")
        pairs = [p for p in pairs if complete(p["pair_id"])]
    x = np.array([[float(feats[p["pair_id"]][c]) for c in columns] for p in pairs])
    y = np.array([label(p) for p in pairs])
    groups = np.array([p["participant_id"] for p in pairs])

    if args.describe:
        fitted = model(0.1).fit(x, y)
        coef = fitted[-1].coef_[0]
        for name, beta in sorted(zip(columns, coef), key=lambda t: -abs(t[1])):
            print(f"{name:12s} {beta:+.3f}")
        return

    p, chosen = nested_cv(x, y, groups, args.repeats, outer_k=5, inner_k=4, seed=args.seed)
    acc = np.mean(np.where(p == 0.5, 0.5, (p > 0.5) == y))
    print(f"block={args.block}  features={len(columns)}  pairs={len(y)}  participants={len(set(groups))}")
    print(f"accuracy {100 * acc:.1f}%   C chosen: " + ", ".join(f"{c}×{chosen.count(c)}" for c in C_GRID))
    suffix = "" if args.block == "all" else f"_{args.block}"
    out = args.out or REPO / "predictions" / f"rubric_regression{suffix}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for pair, prob in zip(pairs, p):
            f.write(json.dumps({"pair_id": pair["pair_id"], "p_memory": float(prob), "score": float(prob) - 0.5}) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
