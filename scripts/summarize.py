"""Every statistic of the release on one page.

  python scripts/summarize.py            # writes results/summary.md (about a minute)

Reads data/, features/ and the frozen predictions in results/predictions/. To use predictions
you have regenerated, pass --predictions predictions/ (file names as in results/predictions/).
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from memrm.data import GENERATORS, REGIMES, REPO, label, load_pairs, load_predictions  # noqa: E402
from memrm.stats import accuracy, cluster_bootstrap_ci, correct, coverage_accuracy, predicted_memory_rate  # noqa: E402
import win_rates  # noqa: E402

PREDICTORS = [("Direct judge (GPT-5.2, one order)", "direct_judge.jsonl", "score_original_order"),
              ("Direct judge (GPT-5.2, both orders)", "direct_judge.jsonl", "score"),
              ("Ours (Response only)", "rubric_model_response.jsonl", "score"),
              ("Ours (Memory only)", "rubric_model_memory.jsonl", "score"),
              ("Ours", "rubric_model.jsonl", "score")]
COVERAGE = (1.0, 0.75, 0.5, 0.25)


def pct(x):
    return f"{100 * x:.1f}"


def ci(lo, hi):
    return f"[{100 * lo:.1f}, {100 * hi:.1f}]"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|"]
    return out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]


def data_section(pairs, counts):
    waves = Counter(p["collection_wave"] for p in pairs)
    lines = ["## 1. Data", "",
             f"- Released clear-preference pairs: **{len(pairs)}** from **{len({p['participant_id'] for p in pairs})}** "
             f"participants ({', '.join(f'{k}: {v}' for k, v in sorted(waves.items()))}).",
             f"- All audited judgments behind the release, including slight preferences and ties: "
             f"**{counts['judgments']}** from **{counts['participants']}** participants (`data/judgment_counts.json`).", ""]
    rows = []
    for key, name in REGIMES.items():
        row = [name] + [sum(p["prompt_variant"] == key and p["generator"] == g for p in pairs) for g in GENERATORS]
        rows.append(row + [sum(row[1:])])
    rows.append(["**All**"] + [sum(p["generator"] == g for p in pairs) for g in GENERATORS] + [len(pairs)])
    return lines + ["Released pairs by prompting regime and generator:", ""] + table(
        ["Prompting regime", *GENERATORS.values(), "All"], rows) + [""]


def all_judgments_section(counts):
    cells = counts["cells"]

    def shares(sel):
        c = Counter()
        for cell in cells:
            if sel(cell):
                c.update({o: cell[o] for o in counts["outcomes"]})
        n = sum(c.values())
        mem, nomem = c["memory_much"] + c["memory_slightly"], c["no_memory_much"] + c["no_memory_slightly"]
        return [n, f"{pct(mem / n)} ({mem})", f"{pct(nomem / n)} ({nomem})", f"{pct(c['tie'] / n)} ({c['tie']})",
                pct(c["memory_much"] / max(c["memory_much"] + c["no_memory_much"], 1))]

    rows = [["**Prompting regime**", "", "", "", "", ""]]
    rows += [[name, *shares(lambda c, k=key: c["prompt_variant"] == k)] for key, name in REGIMES.items()]
    rows += [["**Generator (ordinary prompting)**", "", "", "", "", ""]]
    rows += [[name, *shares(lambda c, g=g: c["generator"] == g and c["prompt_variant"] == "normal")]
             for g, name in GENERATORS.items()]
    rows += [["**Generator (all regimes)**", "", "", "", "", ""]]
    rows += [[name, *shares(lambda c, g=g: c["generator"] == g)] for g, name in GENERATORS.items()]
    rows += [["**All**", *shares(lambda c: True)]]
    return (["## 2. Preferences over all judgments (cf. Table 2)", "",
             "Share of judgments (%, count) preferring each response, including \"slightly better\" and ties. "
             "The last column restricts to clear preferences, i.e. the released pairs.", ""]
            + table(["Group", "Judgments", "Memory", "No memory", "Tie", "Memory, clear only"], rows) + [""])


def clear_section(pairs, n_boot):
    rows = win_rates.table(pairs, n_boot)
    out, section = [], None
    for r in rows:
        if r["section"] != section:
            section = r["section"]
            out.append([f"**{section}**", "", "", "", ""])
        out.append([r["group"], r["pairs"], r["participants"], pct(r["memory_win_rate"]),
                    ci(r["memory_ci_low"], r["memory_ci_high"])])
    return (["## 3. Clear-preference win rates (released pairs)", "",
             "Memory response preferred (%), with 95% participant-clustered bootstrap CIs.", ""]
            + table(["Group", "Pairs", "Participants", "Memory wins", "95% CI"], out) + [""])


def prediction_section(pairs, pred_dir, n_boot):
    preds = {name: load_predictions(pred_dir / f, field) for name, f, field in PREDICTORS}
    keep = set.intersection(*(set(s) for s in preds.values()))
    pairs = [p for p in pairs if p["pair_id"] in keep]
    base = lambda sub: [{"participant_id": p["participant_id"], "generator": p["generator"], "y": label(p)}  # noqa: E731
                        for p in sub]
    majority_class = int(2 * sum(label(p) for p in pairs) > len(pairs))

    def rows_for(name, sub):
        return [{**b, "score": preds[name][p["pair_id"]]} for b, p in zip(base(sub), sub)]

    def majority_correct(sub):
        k = int(2 * sum(label(p) for p in sub) > len(sub))
        return [float(label(p) == k) for p in sub]

    acc_rows, cov_rows = [], []
    by_gen = {g: [p for p in pairs if p["generator"] == g] for g in GENERATORS}
    maj = majority_correct(pairs)
    lo, hi = cluster_bootstrap_ci([{"participant_id": p["participant_id"], "c": c} for p, c in zip(pairs, maj)],
                                  lambda xs: sum(x["c"] for x in xs) / len(xs), n_boot=n_boot)
    acc_rows.append(["Majority", pct(sum(maj) / len(maj)), ci(lo, hi),
                     *[pct(sum(majority_correct(s)) / len(s)) for s in by_gen.values()],
                     f"{pct(sum(label(p) for p in pairs) / len(pairs))} (annotators)"])
    for name, _, _ in PREDICTORS:
        rows = rows_for(name, pairs)
        lo, hi = cluster_bootstrap_ci(rows, accuracy, n_boot=n_boot)
        acc_rows.append([name, pct(accuracy(rows)), ci(lo, hi), *[pct(accuracy(rows_for(name, s))) for s in by_gen.values()],
                         pct(predicted_memory_rate(rows))])
        cov_rows.append([name, *[pct(coverage_accuracy(rows, c)) for c in COVERAGE]])

    def paired(a, b, sub):
        ca = [correct(preds[a][p["pair_id"]], label(p)) for p in sub]
        cb = majority_correct(sub) if b == "Majority" else [correct(preds[b][p["pair_id"]], label(p)) for p in sub]
        items = [{"participant_id": p["participant_id"], "d": x - y} for p, x, y in zip(sub, ca, cb)]
        lo, hi = cluster_bootstrap_ci(items, lambda xs: sum(x["d"] for x in xs) / len(xs), n_boot=n_boot)
        return f"{100 * sum(x['d'] for x in items) / len(items):+.1f} {ci(lo, hi)}"

    comparisons = [("Ours", "Majority"), ("Ours", "Direct judge (GPT-5.2, one order)"),
                   ("Ours", "Ours (Response only)"), ("Ours", "Ours (Memory only)")]
    diff_rows = [[f"{a} − {b}", paired(a, b, pairs), *[paired(a, b, s) for s in by_gen.values()]] for a, b in comparisons]
    gen_names = list(GENERATORS.values())
    return (["## 4. Preference prediction (cf. Tables 3, 5, 7)", "",
             f"{len(pairs)} released pairs with rubric features ({', '.join(f'{n} {len(s)}' for n, s in zip(gen_names, by_gen.values()))}). "
             "Ours = 12-feature rubric reward model, participant-grouped nested cross-validation "
             "(`scripts/rubric_regression.py`). Accuracy (%), 95% participant-clustered CIs.", ""]
            + table(["Predictor", "All", "95% CI", *gen_names, "Picks memory"], acc_rows)
            + ["", "Selective prediction: accuracy (%) on the most confident pairs, ranked by |score|.", ""]
            + table(["Predictor", *[f"{int(100 * c)}%" for c in COVERAGE]], cov_rows)
            + ["", "Paired differences in accuracy (percentage points, 95% CI):", ""]
            + table(["Comparison", "All", *gen_names], diff_rows) + [""]), pairs


def coefficient_section(pairs):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    spec = json.loads((REPO / "features" / "feature_spec.json").read_text())["features"]
    with open(REPO / "features" / "rubric_features.csv", newline="") as f:
        feats = {r["pair_id"]: r for r in csv.DictReader(f)}
    x = [[float(feats[p["pair_id"]][s["name"]]) for s in spec] for p in pairs]
    y = [label(p) for p in pairs]
    coef = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=10_000)).fit(x, y)[-1].coef_[0]
    order = sorted(zip(spec, coef), key=lambda t: -abs(t[1]))
    rows = [[i + 1, s["name"], s["definition"].split(":")[0], f"{c:+.3f}", "Yes" if s["block"] == "memory" else "No"]
            for i, (s, c) in enumerate(order)]
    return (["## 5. Feature coefficients (cf. Tables 4, 6)", "",
             "Standardised L2 logistic-regression coefficients of a descriptive fit to all evaluated pairs (C = 0.1). "
             "Positive values favour the memory-conditioned response.", ""]
            + table(["Rank", "Feature", "Definition", "β", "Memory feature"], rows) + [""])


def paper_run_section():
    path = REPO / "results" / "paper_run" / "induced_rubric_model.json"
    if not path.exists():
        return []
    run = json.loads(path.read_text())
    short = {g.split("/")[-1]: name for g, name in GENERATORS.items()}
    gens = [g for g in short if g in run["accuracy_by_generator"]]
    rows = [[name, v["accuracy"], ci(v["ci95"][0] / 100, v["ci95"][1] / 100),
             *[run["accuracy_by_generator"][g].get(name, "") for g in gens],
             *run["coverage_accuracy"].get(name, {}).values()] for name, v in run["accuracy"].items()]
    return (["## 6. Induced-rubric model, paper run", "",
             f"Rubrics induced and scored by {run['model']} within each outer fold, on the paper's {run['pairs']} "
             f"pairs before the release audit (majority {run['majority']}%). Rubrics: `results/paper_run/`.", ""]
            + table(["Features", "All", "95% CI", *[short[g] for g in gens], "Cov. 100%", "75%", "50%", "25%"],
                    [r + [""] * (3 + len(gens) + 4 - len(r)) for r in rows]) + [""])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", type=Path, default=REPO / "results" / "predictions")
    parser.add_argument("--out", type=Path, default=REPO / "results" / "summary.md")
    parser.add_argument("--n-boot", type=int, default=10_000)
    args = parser.parse_args()
    pairs = load_pairs()
    counts = json.loads((REPO / "data" / "judgment_counts.json").read_text())
    prediction, evaluated = prediction_section(pairs, args.predictions, args.n_boot)
    lines = (["# Release statistics", "", "Generated by `python scripts/summarize.py`.", ""]
             + data_section(pairs, counts) + all_judgments_section(counts) + clear_section(pairs, args.n_boot)
             + prediction + coefficient_section(evaluated) + paper_run_section())
    args.out.write_text("\n".join(lines))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
