# Memory-Aware Reward Modeling

Data and code for [Memory-Aware Reward Modeling: Learning Human Preferences for LLM Memory Use](https://arxiv.org/abs/2610.31272).

Participants compared two answers to their own ChatGPT queries. The same model wrote both
answers, one with the participant's retrieved memories in context and one without them.

## Data

`data/memory_preferences.jsonl` holds 262 clear-preference pairs from 73 participants
(fields: [`data/README.md`](data/README.md)). The release re-applies the collection audit
and de-identification, so its numbers differ slightly from the paper's 291 pairs.

| Generator | Pairs | Memory response preferred |
|---|---:|---:|
| GPT-5.2 | 106 | 54.7% |
| Gemini-3.1-Pro | 117 | 27.4% |
| Claude-Opus-5 | 39 | 74.4% |
| **All** | **262** | **45.4%** |

```python
from memrm.data import load_pairs, memory_response, no_memory_response, label
pairs = load_pairs()   # label(pair) == 1 if the memory-conditioned response was preferred
```

## Results

Preference-prediction accuracy (%) on the 253 released pairs with rubric features
(participant-grouped nested cross-validation; full table in `results/prediction_accuracy.md`).

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

## De-identification

- Recruitment IDs, platform IDs, names, timestamps and other operational metadata were removed,
  and participants were given random IDs.
- Each pair includes only the nine memories retrieved for it, not the full profile.
- Contact details are masked. Locations, venues, organisations and personal names are replaced
  by typed placeholders (`[PERSON_1]`, `[LOCATION_1]`, …), consistent within each participant.
- Every memory, query and note was reviewed for special-category data under GDPR (health,
  religion, ethnic origin, political opinion, sex life), offence data and information about
  third parties. Such content was removed or generalised.
- Responses were not rewritten. If a response relied on removed information, the whole pair
  was withheld.

Please do not attempt to identify participants. To report identifying content or a withdrawal
request, contact yd358@cam.ac.uk.

## License and citation

Code: MIT. Data and prompts: CC BY-NC 4.0.

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
