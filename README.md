# Memory-Aware Reward Modeling

Data and code for [Memory-Aware Reward Modeling: Learning Human Preferences for LLM Memory Use](https://arxiv.org/abs/2610.31272).

Participants compared two answers to their own ChatGPT queries. The same model wrote both
answers, one with the participant's retrieved memories in context and one without them.

## Data

`data/memory_preferences.jsonl` holds 262 clear-preference pairs from 73 participants
(fields: [`data/README.md`](data/README.md)). The release re-applies the collection audit
and de-identification, so its numbers differ slightly from the paper's 291 pairs.

Over all 658 judgments, including slight preferences and ties (`data/judgment_counts.json`):

| Prompting regime | Memory | No memory | Tie |
|---|---:|---:|---:|
| Conservative | 40.9% | 41.7% | 17.3% |
| Ordinary | 39.0% | 44.5% | 16.5% |
| Aggressive | 35.1% | 46.6% | 18.3% |

```python
from memrm.data import load_pairs, memory_response, no_memory_response, label
pairs = load_pairs()   # label(pair) == 1 if the memory-conditioned response was preferred
```

## Results

All statistics of the release are on one page: [`results/summary.md`](results/summary.md).

Preference-prediction accuracy (%) on the 253 released pairs with rubric features
(participant-grouped nested cross-validation).

| Predictor | All | GPT-5.2 | Gemini | Claude |
|---|---:|---:|---:|---:|
| Majority | 54.5 | 55.1 | 72.4 | 74.4 |
| Direct judge (GPT-5.2) | 59.3 | 56.1 | 58.6 | 69.2 |
| Ours (Response only) | 68.8 | 62.2 | 74.1 | 69.2 |
| Ours (Memory only) | 63.6 | 58.2 | 67.2 | 66.7 |
| **Ours** | **74.7** | **67.3** | **75.9** | **89.7** |

## Code

```bash
pip install -r requirements.txt
python scripts/summarize.py                                   # every statistic -> results/summary.md
python scripts/win_rates.py                                   # memory vs no-memory win rates
python scripts/rubric_regression.py --features features/rubric_features.csv [--block memory|response]
python scripts/evaluate.py --pairs-in features/rubric_features.csv predictions/rubric_regression.jsonl
```

`features/rubric_features.csv` holds the twelve feature scores behind the results above
(definitions in `features/feature_spec.json`). To score them again:

```bash
python scripts/score_features.py --pass all --execute         # GPT-5.2 and qwen-max; needs API keys
python scripts/build_features.py
```

Baselines: `scripts/run_llm_judge.py` (direct and rubric GPT-5.2 judges) and
`scripts/run_skywork.py` (Skywork-Reward-V2, GPU). `prompts/` contains the collection,
generation, screening, judge, feature-scoring and rubric-induction prompts.
`results/paper_run/` has the induced rubrics and results of the paper run.

```bibtex
@article{dong2026memoryaware,
  title   = {Memory-Aware Reward Modeling: Learning Human Preferences for LLM Memory Use},
  author  = {Dong, Yijiang River and He, Zexue and Yu, Cheng and Hu, Tiancheng and Ryan, Michael J and
             Batzner, Jan and Hui, Zheng and Wu, Shaolong and Papakyriakopoulos, Orestis and
             Pentland, Alex and Collier, Nigel and Pei, Jiaxin},
  journal = {arXiv preprint arXiv:2610.31272},
  year    = {2026}
}
```
