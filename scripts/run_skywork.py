"""Pretrained Skywork-Reward-V2 baseline (Appendix D.2).

Each candidate response is scored independently. The user turn holds the query and, unless
--no-memory is given, the retrieved memories under [User Query] and [Relevant Memories];
the assistant turn holds the response. The tokenizer's chat template formats the turns and
the sequence-classification head returns a scalar reward. The higher-reward response is
predicted to be preferred; no parameters are updated.

  python scripts/run_skywork.py                       # memories in context
  python scripts/run_skywork.py --no-memory           # query only
  python scripts/run_skywork.py --model Skywork/Skywork-Reward-V2-Qwen3-0.6B

Writes predictions/skywork_<with|no>_memory.jsonl with score = r(memory response) - r(no-memory response).
The 8B checkpoint needs a GPU with about 20 GB of memory in bfloat16.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from memrm.data import DATA, REPO, load_pairs, memory_response, memory_texts, no_memory_response  # noqa: E402


def user_turn(pair, with_memory):
    text = f"[User Query]\n{pair['query']['text']}"
    if with_memory:
        memories = "\n".join(f"{i}. {m}" for i, m in enumerate(memory_texts(pair), 1))
        text += f"\n\n[Relevant Memories]\n{memories}"
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Skywork/Skywork-Reward-V2-Llama-3.1-8B")
    parser.add_argument("--no-memory", action="store_true", help="omit the memories from the input")
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--max-length", type=int, default=16384)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, num_labels=1).to(device).eval()

    @torch.no_grad()
    def reward(user, assistant):
        conv = [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
        ids = tokenizer.apply_chat_template(conv, tokenize=True, return_tensors="pt",
                                            truncation=True, max_length=args.max_length).to(device)
        return model(ids).logits[0][0].item()

    pairs = load_pairs(args.data)
    tag = "no_memory" if args.no_memory else "with_memory"
    out = args.out or REPO / "predictions" / f"skywork_{tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for i, pair in enumerate(pairs, 1):
            user = user_turn(pair, not args.no_memory)
            r_mem, r_nomem = reward(user, memory_response(pair)), reward(user, no_memory_response(pair))
            f.write(json.dumps({"pair_id": pair["pair_id"], "reward_memory_response": r_mem,
                                "reward_no_memory_response": r_nomem, "score": r_mem - r_nomem}) + "\n")
            if i % 25 == 0:
                print(f"{i}/{len(pairs)}", flush=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
