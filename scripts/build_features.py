"""Combine the LLM scores from scripts/score_features.py with the text features into one table.

  python scripts/build_features.py            # writes features/rubric_features_rescored.csv

Each feature is signed so that larger values favour the memory-conditioned response:
  HEAD, LEN, AXIS, OPT    memrm/features.py
  PROP                    aug18 R2, oriented towards the memory-conditioned response
  REL                     memory_fit relevance / 10
  COMMIT ... REGISTER     aug24 A1, A2, A4, A5, A6, A7, averaged over both A/B orders and oriented
The frozen paper scores are in features/rubric_features.csv; this script does not overwrite them.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from memrm.data import DATA, REPO, load_pairs, memory_response, no_memory_response  # noqa: E402
from memrm.features import text_features  # noqa: E402

AUG24 = {"COMMIT": "A1", "BREADTH": "A2", "ASSUMP": "A4", "COST": "A5", "CLOSURE": "A6", "REGISTER": "A7"}


def load_calls(path):
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            out[(r["pair_id"], r["order"])] = r["scores"]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--runs", type=Path, default=REPO / "runs" / "features")
    parser.add_argument("--out", type=Path, default=REPO / "features" / "rubric_features_rescored.csv")
    args = parser.parse_args()
    columns = [f["name"] for f in json.loads((REPO / "features" / "feature_spec.json").read_text())["features"]]
    aug18, fit, aug24 = (load_calls(args.runs / p / "calls.jsonl") for p in ("aug18", "memory_fit", "aug24"))
    rows, incomplete = [], 0
    for pair in load_pairs(args.data):
        pid = pair["pair_id"]
        sign = 1.0 if pair["memory_side"] == "A" else -1.0
        row = {"pair_id": pid, **text_features(memory_response(pair), no_memory_response(pair))}
        if (pid, "AB") in aug18:
            row["PROP"] = sign * aug18[(pid, "AB")]["R2"]
        if (pid, "AB") in fit:
            row["REL"] = fit[(pid, "AB")]["relevance"] / 10
        if (pid, "AB") in aug24 and (pid, "BA") in aug24:
            for name, dim in AUG24.items():
                # the BA call saw the responses swapped; map its score back to released A
                row[name] = sign * (aug24[(pid, "AB")][dim] - aug24[(pid, "BA")][dim]) / 2
        incomplete += any(c not in row for c in columns)
        rows.append(row)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pair_id"] + columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()})
    print(f"wrote {args.out}: {len(rows)} pairs, {incomplete} with missing LLM scores")


if __name__ == "__main__":
    main()
