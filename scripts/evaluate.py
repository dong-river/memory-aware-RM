"""Preference-prediction accuracy (cf. Tables 3, 5 and 7 of the paper).

Each predictions file is JSONL with one {"pair_id": ..., "<field>": float} per line, where
the score is oriented towards the memory-conditioned response (> 0 predicts that the
participant preferred it). Exact zeros count as ties and receive half credit.

  python scripts/evaluate.py predictions/direct.jsonl predictions/rubric.jsonl \
      predictions/skywork_with_memory.jsonl predictions/rubric_regression.jsonl

Use NAME=PATH to label a file, and PATH:FIELD to read a field other than "score"
(e.g. predictions/direct.jsonl:score_original_order for single-order judging).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from memrm.data import DATA, GENERATORS, label, load_pairs, load_predictions  # noqa: E402
from memrm.stats import (accuracy, cluster_bootstrap_ci, coverage_accuracy, majority_accuracy,  # noqa: E402
                         predicted_memory_rate)

COVERAGE = (1.0, 0.75, 0.5, 0.25)


def parse_spec(spec):
    name, _, rest = spec.rpartition("=") if "=" in spec else ("", "", spec)
    path, _, field = rest.partition(":")
    return name or Path(path).stem + (f":{field}" if field else ""), Path(path), field or "score"


def rows_for(pairs, scores, name):
    missing = [p["pair_id"] for p in pairs if p["pair_id"] not in scores]
    if missing:
        raise SystemExit(f"{name}: no prediction for {len(missing)} pairs (e.g. {missing[0]}); "
                         "evaluate on the full release or pass --allow-missing")
    return [{"participant_id": p["participant_id"], "generator": p["generator"],
             "y": label(p), "score": scores[p["pair_id"]]} for p in pairs]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("predictions", nargs="*", help="[NAME=]PATH[:FIELD]")
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--allow-missing", action="store_true",
                        help="evaluate each predictor on the pairs it covers (changes the denominator)")
    parser.add_argument("--pairs-in", type=Path,
                        help="restrict every row, majority included, to the pair_ids in this CSV or JSONL file")
    parser.add_argument("--n-boot", type=int, default=10_000)
    args = parser.parse_args()
    pairs = load_pairs(args.data)
    if args.pairs_in:
        if args.pairs_in.suffix == ".csv":
            with open(args.pairs_in, newline="") as f:
                keep = {r["pair_id"] for r in csv.DictReader(f)}
        else:
            keep = {json.loads(line)["pair_id"] for line in args.pairs_in.read_text().splitlines() if line.strip()}
        pairs = [p for p in pairs if p["pair_id"] in keep]
    gens = list(GENERATORS)

    def header(cols):
        print("| " + " | ".join(cols) + " |")
        print("|" + "|".join(["---"] + ["---:"] * (len(cols) - 1)) + "|")

    counts = ", ".join("{} {}".format(GENERATORS[g], sum(p["generator"] == g for p in pairs)) for g in gens)
    print(f"Pairs: {len(pairs)}  ({counts})\n")
    print("Accuracy (%). Majority = always predicting the more frequent outcome within each column.\n")
    header(["Predictor", "All", "95% CI", *[GENERATORS[g] for g in gens], "Picks memory", "n"])
    base = [{"participant_id": p["participant_id"], "generator": p["generator"], "y": label(p), "score": 0.0}
            for p in pairs]
    by_gen = [100 * majority_accuracy([r for r in base if r["generator"] == g]) for g in gens]
    lo, hi = cluster_bootstrap_ci(base, majority_accuracy, n_boot=args.n_boot)
    human = 100 * sum(r["y"] for r in base) / len(base)
    print(f"| Majority | {100 * majority_accuracy(base):.1f} | [{100 * lo:.1f}, {100 * hi:.1f}] | "
          + " | ".join(f"{a:.1f}" for a in by_gen) + f" | annotators: {human:.1f}% | {len(base)} |")

    coverage_rows = []
    for spec in args.predictions:
        name, path, field = parse_spec(spec)
        scores = load_predictions(path, field)
        subset = [p for p in pairs if p["pair_id"] in scores] if args.allow_missing else pairs
        rows = rows_for(subset, scores, name)
        lo, hi = cluster_bootstrap_ci(rows, accuracy, n_boot=args.n_boot)
        per_gen = []
        for g in gens:
            sub = [r for r in rows if r["generator"] == g]
            per_gen.append(f"{100 * accuracy(sub):.1f}" if sub else "–")
        print(f"| {name} | {100 * accuracy(rows):.1f} | [{100 * lo:.1f}, {100 * hi:.1f}] | " + " | ".join(per_gen)
              + f" | {100 * predicted_memory_rate(rows):.1f}% | {len(rows)} |")
        coverage_rows.append((name, [100 * coverage_accuracy(rows, c) for c in COVERAGE]))

    if coverage_rows:
        print("\nSelective prediction: accuracy (%) on the most confident pairs, ranked by |score|.\n")
        header(["Predictor", *[f"{int(100 * c)}%" for c in COVERAGE]])
        for name, accs in coverage_rows:
            print(f"| {name} | " + " | ".join(f"{a:.1f}" for a in accs) + " |")


if __name__ == "__main__":
    main()
