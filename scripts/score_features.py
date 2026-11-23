"""LLM scoring passes behind the eight judged features of the rubric reward model (Table 6).

  pass        model (paper)   features          input
  aug18       GPT-5.2         PROP (R2)         query, memories, both responses (one order)
  memory_fit  GPT-5.2         REL (relevance)   query and memories only, no responses
  aug24       qwen-max        COMMIT, BREADTH,  query and both responses, no memories,
                              ASSUMP, COST,     scored in both A/B orders
                              CLOSURE, REGISTER

The prompts are in prompts/features/. Calls are cached in runs/features/<pass>/calls.jsonl,
so an interrupted run resumes where it stopped. Without --execute the script only reports
how many calls it would make. Combine the scores with scripts/build_features.py.

  export OPENAI_API_KEY=...                      # aug18, memory_fit
  python scripts/score_features.py --pass aug18 --execute
  python scripts/score_features.py --pass memory_fit --execute
  export DASHSCOPE_API_KEY=...                   # aug24 (qwen-max, OpenAI-compatible endpoint)
  python scripts/score_features.py --pass aug24 --execute

The released feature table (features/rubric_features.csv) holds the scores from the paper's
run, made on the text before de-identification. One few-shot example in the aug18 prompt was
adapted from a participant's memories; the released prompt replaces that detail with a
fictional one, so re-scored values can differ slightly.
"""
import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from memrm.data import DATA, REPO, load_pairs  # noqa: E402

PROMPTS = REPO / "prompts" / "features"
PASSES = {
    "aug18": {"model": "gpt-5.2", "dims": ["R1", "R2", "R3", "R5", "R6", "R7", "R8", "R9", "R11", "R12", "R13"],
              "range": (-5, 5), "orders": ("AB",), "key_env": "OPENAI_API_KEY", "base_url": None,
              "params": {"max_completion_tokens": 1200, "reasoning_effort": "low"}},
    "memory_fit": {"model": "gpt-5.2", "dims": ["benefit", "risk", "relevance", "needs_user_facts",
                                                "generic_sufficient", "intrusiveness"],
                   "range": (0, 10), "orders": ("AB",), "key_env": "OPENAI_API_KEY", "base_url": None,
                   "params": {"max_completion_tokens": 300, "reasoning_effort": "low"}},
    "aug24": {"model": "qwen-max", "dims": ["A1", "A2", "A3", "A4", "A5", "A6", "A7"],
              "range": (-1e9, 1e9), "orders": ("AB", "BA"), "key_env": "DASHSCOPE_API_KEY",
              "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
              "params": {"max_tokens": 300}},
}


def memories(pair, keep_removed):
    return [m for m in pair["memories"] if keep_removed or not m.get("redacted")]


def responses(pair, order):
    a, b = pair["response_a"]["text"], pair["response_b"]["text"]
    return (b, a) if order == "BA" else (a, b)


def messages_aug18(pair, order, keep_removed):
    lines = [f"  - {m['text']}" + (f" (similarity={m['similarity']:.3f})" if m.get("similarity") else "")
             for m in memories(pair, keep_removed)]
    a, b = responses(pair, order)
    user = (f"## User Query\n{pair['query']['text']}\n\n## Retrieved Memories\n"
            f"{chr(10).join(lines) if lines else '  (none)'}\n\n## Response A\n{a}\n\n## Response B\n{b}")
    few_shot = json.loads((PROMPTS / "rubric_aug18_fewshot.json").read_text())
    return ([{"role": "system", "content": (PROMPTS / "rubric_aug18_system.txt").read_text()}]
            + few_shot + [{"role": "user", "content": user}])


def messages_memory_fit(pair, order, keep_removed):
    lines = [f"  - {m['text']}" + (f" [{m['type']}]" if m.get("type") else "") for m in memories(pair, keep_removed)]
    user = (f"## User question\n{pair['query']['text']}\n\n## Retrieved memories about this user\n"
            f"{chr(10).join(lines) if lines else '  (none)'}\n\nRate the situation.")
    return [{"role": "system", "content": (PROMPTS / "memory_fit_system.txt").read_text()},
            {"role": "user", "content": user}]


def messages_aug24(pair, order, keep_removed):
    a, b = responses(pair, order)
    user = f"## Question\n{pair['query']['text']}\n\n## Response A\n{a[:3500]}\n\n## Response B\n{b[:3500]}\n\nScore each axis."
    return [{"role": "system", "content": (PROMPTS / "rubric_aug24_system.txt").read_text()},
            {"role": "user", "content": user}]


BUILDERS = {"aug18": messages_aug18, "memory_fit": messages_memory_fit, "aug24": messages_aug24}


def parse(text):
    text = re.sub(r"```(?:json)?\s*", "", text or "").strip().rstrip("`").strip()
    for candidate in (text, (re.search(r"\{.*\}", text, re.DOTALL) or [None])[0]):
        try:
            return json.loads(candidate)
        except (TypeError, json.JSONDecodeError):
            continue
    return None


def extract(parsed, spec):
    scores = (parsed or {}).get("scores", parsed) or {}
    lo, hi = spec["range"]
    out = {}
    for d in spec["dims"]:
        v = scores.get(d)
        if not isinstance(v, (int, float)) or not lo <= v <= hi:
            return None
        out[d] = float(v)
    return out


async def score(pass_name, pairs, args):
    from openai import AsyncOpenAI
    spec = PASSES[pass_name]
    key = os.getenv(args.api_key_env or spec["key_env"])
    if not key:
        raise SystemExit(f"{args.api_key_env or spec['key_env']} is not set")
    client = AsyncOpenAI(api_key=key, base_url=args.base_url or spec["base_url"])
    model = args.model or spec["model"]
    run_dir = args.runs / pass_name
    run_dir.mkdir(parents=True, exist_ok=True)
    calls_path = run_dir / "calls.jsonl"
    done = set()
    if calls_path.exists():
        done = {(r["pair_id"], r["order"]) for r in map(json.loads, calls_path.read_text().splitlines())}
    todo = [(p, o) for p in pairs for o in spec["orders"] if (p["pair_id"], o) not in done]
    sem = asyncio.Semaphore(args.concurrency)

    async def one(pair, order):
        msgs = BUILDERS[pass_name](pair, order, args.keep_removed_memories)
        async with sem:
            for attempt in range(3):
                try:
                    r = await client.chat.completions.create(model=model, messages=msgs,
                                                             response_format={"type": "json_object"}, **spec["params"])
                    scores = extract(parse(r.choices[0].message.content), spec)
                    if scores:
                        return {"pair_id": pair["pair_id"], "order": order, "model": model, "scores": scores,
                                "input_sha256": hashlib.sha256(json.dumps(msgs).encode()).hexdigest()}
                except Exception as e:  # noqa: BLE001
                    print(f"  [retry {attempt + 1}] {pair['pair_id']} {order}: {str(e)[:120]}")
                    await asyncio.sleep(2 ** attempt)
        print(f"  [failed] {pair['pair_id']} {order}")
        return None

    with open(calls_path, "a") as f:
        for i, coro in enumerate(asyncio.as_completed([one(p, o) for p, o in todo]), 1):
            rec = await coro
            if rec:
                f.write(json.dumps(rec) + "\n")
                f.flush()
            if i % 25 == 0 or i == len(todo):
                print(f"  {pass_name}: {i}/{len(todo)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pass", dest="pass_name", choices=list(PASSES) + ["all"], required=True)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--runs", type=Path, default=REPO / "runs" / "features")
    parser.add_argument("--model", help="override the pass's default model")
    parser.add_argument("--base-url", help="override the API base URL")
    parser.add_argument("--api-key-env", help="environment variable holding the API key")
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--keep-removed-memories", action="store_true",
                        help="show redacted memories as their placeholder instead of dropping them")
    parser.add_argument("--execute", action="store_true", help="make the API calls (otherwise dry run)")
    args = parser.parse_args()
    pairs = load_pairs(args.data)
    for name in (PASSES if args.pass_name == "all" else [args.pass_name]):
        n = len(pairs) * len(PASSES[name]["orders"])
        if not args.execute:
            print(f"{name}: {n} calls to {args.model or PASSES[name]['model']} (dry run; add --execute)")
            continue
        asyncio.run(score(name, pairs, args))


if __name__ == "__main__":
    main()
