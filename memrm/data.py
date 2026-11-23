"""Loading and small helpers for the released preference pairs."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "memory_preferences.jsonl"
REMOVED_MEMORY = "[removed: sensitive personal information]"

# prompt_variant value -> name used in the paper (Section 3.2, Appendix E.2)
REGIMES = {
    "least": "Conservative",
    "normal": "Ordinary",
    "aggressive": "Aggressive",
    "generic_then_personalized": "Generic-then-personalized",
}
GENERATORS = {
    "gpt-5.2": "GPT-5.2",
    "google/gemini-3.1-pro-preview": "Gemini-3.1-Pro",
    "anthropic/claude-opus-5": "Claude-Opus-5",
}


def load_pairs(path=DATA):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def label(pair):
    """1 if the participant preferred the memory-conditioned response, else 0."""
    return int(pair["label"]["memory_preferred"])


def memory_response(pair):
    return pair["response_a" if pair["memory_side"] == "A" else "response_b"]["text"]


def no_memory_response(pair):
    return pair["response_b" if pair["memory_side"] == "A" else "response_a"]["text"]


def memory_texts(pair, keep_removed=False):
    """The retrieved memories shown to the generator, in retrieval order.

    Memories withheld during de-identification carry REMOVED_MEMORY as text; they are
    dropped unless keep_removed is set.
    """
    return [m["text"] for m in pair["memories"] if keep_removed or not m.get("redacted")]


def regime(pair):
    return REGIMES.get(pair["prompt_variant"], pair["prompt_variant"])


def generator(pair):
    return GENERATORS.get(pair["generator"], pair["generator"])


def load_predictions(path, field="score"):
    """Read a predictions JSONL file: one {"pair_id": ..., <field>: float} per line.

    Scores are oriented towards the memory-conditioned response: > 0 predicts that the
    participant prefers it, < 0 the response without memory, 0 is a tie.
    """
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                out[row["pair_id"]] = float(row[field])
    return out
