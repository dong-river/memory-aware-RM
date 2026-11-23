"""Memory vs no-memory win rates on the released pairs (cf. Table 2 of the paper).

The release contains clear-preference pairs only (A/B "much better"), so there are no ties
and the two win rates sum to 100%. Table 2 of the paper also counts "slightly better" and
"about the same" judgments, which are not released, so its percentages are not directly
comparable with these.

  python scripts/win_rates.py            # writes results/win_rates.{md,csv,pdf,png}
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from memrm.data import DATA, GENERATORS, REGIMES, REPO, label, load_pairs  # noqa: E402
from memrm.stats import cluster_bootstrap_ci  # noqa: E402

MEM, NOMEM = "#2a78d6", "#eb6834"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"


def groups(pairs):
    yield "Overall", "All pairs", pairs
    for key, name in REGIMES.items():
        yield "Prompting regime", name, [p for p in pairs if p["prompt_variant"] == key]
    for key, name in GENERATORS.items():
        yield "Generator (ordinary prompting)", name, [
            p for p in pairs if p["generator"] == key and p["prompt_variant"] == "normal"]
    for key, name in GENERATORS.items():
        yield "Generator (all regimes)", name, [p for p in pairs if p["generator"] == key]


def win_rate(pairs):
    return sum(label(p) for p in pairs) / len(pairs)


def table(pairs, n_boot):
    rows = []
    for section, name, subset in groups(pairs):
        if not subset:
            continue
        lo, hi = cluster_bootstrap_ci(subset, win_rate, n_boot=n_boot)
        rate = win_rate(subset)
        rows.append({"section": section, "group": name, "pairs": len(subset),
                     "participants": len({p["participant_id"] for p in subset}),
                     "memory_win_rate": rate, "no_memory_win_rate": 1 - rate,
                     "memory_ci_low": lo, "memory_ci_high": hi})
    return rows


def write_markdown(rows, path):
    lines = ["| Group | Pairs | Participants | Memory wins | 95% CI | No memory wins |",
             "|---|---:|---:|---:|---|---:|"]
    section = None
    for r in rows:
        if r["section"] != section:
            section = r["section"]
            lines.append(f"| **{section}** | | | | | |")
        lines.append(f"| {r['group']} | {r['pairs']} | {r['participants']} | {100 * r['memory_win_rate']:.1f}% | "
                     f"[{100 * r['memory_ci_low']:.1f}, {100 * r['memory_ci_high']:.1f}] | "
                     f"{100 * r['no_memory_win_rate']:.1f}% |")
    path.write_text("\n".join(lines) + "\n")


def plot(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    rows = [r for r in rows if r["section"] != "Generator (all regimes)"]
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9,
                         "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]})
    ys, y, prev = [], 0.0, None
    for r in rows:
        if prev is not None and r["section"] != prev:
            y += 0.55
        ys.append(y)
        y += 1.0
        prev = r["section"]
    fig, ax = plt.subplots(figsize=(6.8, 0.42 * y + 1.0))
    h, gap = 0.62, 0.004
    for r, yy in zip(rows, ys):
        m = r["memory_win_rate"]
        ax.barh(yy, m - gap, left=0, height=h, color=MEM, linewidth=0)
        ax.barh(yy, 1 - m - gap, left=m + gap, height=h, color=NOMEM, linewidth=0)
        ax.text(0.015, yy, f"{100 * m:.1f}%", va="center", ha="left", color="white", fontsize=8.5, weight="bold")
        x_right = r["memory_ci_high"] + 0.012 if r["memory_ci_high"] > 0.88 else 0.985
        ax.text(x_right, yy, f"{100 * (1 - m):.1f}%", va="center", ha="left" if x_right < 0.98 else "right",
                color="white", fontsize=8.5, weight="bold")
        ax.plot([r["memory_ci_low"], r["memory_ci_high"]], [yy, yy], color=INK, lw=1.1)
        for x in (r["memory_ci_low"], r["memory_ci_high"]):
            ax.plot([x, x], [yy - 0.13, yy + 0.13], color=INK, lw=1.1)
        ax.text(1.02, yy, f"n={r['pairs']}", va="center", ha="left", color=INK2, fontsize=8)
    ax.axvline(0.5, color=INK, lw=0.8, ls=(0, (3, 2)), zorder=0.5)
    ax.set_yticks(ys)
    ax.set_yticklabels([r["group"] + (" (ordinary)" if r["section"].startswith("Generator (ordinary") else "")
                        for r in rows], color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], color=MUTED)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("Share of clear-preference pairs won", color=INK2)
    ax.legend(handles=[Patch(color=MEM, label="With memory wins"), Patch(color=NOMEM, label="Without memory wins"),
                       Line2D([0], [0], color=INK, lw=1.1, label="95% CI (participant-clustered)")],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=220, bbox_inches="tight", facecolor="white")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--out", type=Path, default=REPO / "results" / "win_rates")
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    rows = table(load_pairs(args.data), args.n_boot)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out.with_suffix(".csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()} for r in rows)
    write_markdown(rows, args.out.with_suffix(".md"))
    if not args.no_plot:
        plot(rows, args.out)
    print(args.out.with_suffix(".md").read_text())


if __name__ == "__main__":
    main()
