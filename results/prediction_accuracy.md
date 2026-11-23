Pairs: 253  (GPT-5.2 98, Gemini-3.1-Pro 116, Claude-Opus-5 39)

Accuracy (%). Majority = always predicting the more frequent outcome within each column.

| Predictor | All | 95% CI | GPT-5.2 | Gemini-3.1-Pro | Claude-Opus-5 | Picks memory | n |
|---|---:|---:|---:|---:|---:|---:|---:|
| Majority | 54.5 | [50.2, 62.8] | 55.1 | 72.4 | 74.4 | annotators: 45.5% | 253 |
| Direct judge | 59.3 | [50.2, 68.2] | 56.1 | 58.6 | 69.2 | 68.0% | 253 |
| Direct judge (both orders) | 59.7 | [51.9, 67.4] | 58.2 | 59.1 | 65.4 | 66.8% | 253 |
| Ours (Response only) | 68.8 | [62.5, 74.7] | 62.2 | 74.1 | 69.2 | 41.1% | 253 |
| Ours (Memory only) | 63.6 | [58.0, 69.3] | 58.2 | 67.2 | 66.7 | 39.1% | 253 |
| Ours | 74.7 | [68.3, 80.6] | 67.3 | 75.9 | 89.7 | 46.2% | 253 |

Selective prediction: accuracy (%) on the most confident pairs, ranked by |score|.

| Predictor | 100% | 75% | 50% | 25% |
|---|---:|---:|---:|---:|
| Direct judge | 59.3 | 63.7 | 62.7 | 71.4 |
| Direct judge (both orders) | 59.7 | 62.6 | 65.9 | 74.6 |
| Ours (Response only) | 68.8 | 75.3 | 81.0 | 85.7 |
| Ours (Memory only) | 63.6 | 67.4 | 70.6 | 73.0 |
| Ours | 74.7 | 80.0 | 86.5 | 93.7 |
