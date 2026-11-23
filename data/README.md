# Data card: memory-preference pairs

`memory_preferences.jsonl` holds **262 clear-preference comparisons from 73 participants**.
Each comparison is one query answered twice by the same generator, once with the participant's
retrieved memories in context and once without, together with the participant's judgment of
which answer they would rather receive. Participants judged answers to their own queries,
grounded in their own memories. `metadata.json` gives the counts, processing summary and the
file's SHA-256.

| | Pairs | Participants | Memory response preferred |
|---|---:|---:|---:|
| GPT-5.2 | 106 | 51 | 58 (54.7%) |
| Gemini-3.1-Pro | 117 | 48 | 32 (27.4%) |
| Claude-Opus-5 | 39 | 9 | 29 (74.4%) |
| **All** | **262** | **73** | **119 (45.4%)** |

## Collection

Participants were recruited on Prolific (about 30 minutes, USD 14/hour) under a study approved
by an institutional research ethics committee and gave informed consent. Each participant ran
an extraction prompt in their own ChatGPT account (`../prompts/collection/`) to produce an
editable profile and a list of recent queries, and typed or dictated events from the preceding
month. Full chat histories and attachments were never uploaded. For each query, nine memories
were retrieved with `text-embedding-3-small`, and one generator wrote a response with the
memories (under one of the prompting regimes in `../prompts/generation/`) and a response without
them. Participants chose between the two on a five-point scale and marked which memories were
relevant. See Section 3 and Appendix A of the paper.

## Selection

1. The collection audit of Section 3.3, applied in order: incomplete response pair, failed
   participant attention check, failed item sentinel check, selected memory IDs not among the
   retrieved ones, preference inconsistent with the recorded condition.
2. Comparisons between a memory-conditioned and a no-memory response.
3. Clear preferences only (`a_much` / `b_much`). "Slightly better" and "about the same"
   judgments are not released as records. Test–retest agreement was much lower for slight
   preferences (Section 3.4), so the paper trains and evaluates its predictors on clear
   preferences only. `judgment_counts.json` gives the counts of all 658 judgments (84
   participants) by prompting regime, generator and preference strength, after the same
   exclusions; its `*_much` counts are exactly the released pairs.
4. Responses from a fourth generator that was piloted in the second wave are excluded.
5. De-identification (below) withheld 27 further pairs, and one further participant (6 pairs)
   was excluded by the study team.

## Record fields

| Field | Meaning |
|---|---|
| `pair_id` | `<participant_id>_<query id>` |
| `participant_id` | random study identifier (`U01`–`U78`; IDs of fully withheld participants are unused) |
| `collection_wave` | `wave1` (GPT-5.2, Gemini-3.1-Pro) or `wave2` (adds Claude-Opus-5) |
| `generator` | model that wrote both responses |
| `prompt_variant` | memory-use instruction for the memory-conditioned response: `least` = conservative, `normal` = ordinary, `aggressive`, `generic_then_personalized` (texts in `../prompts/generation/`) |
| `pipeline_version` | generation pipeline tag (`v9_2` in wave 2, null in wave 1) |
| `query` | `id`; `text`; `source`: `recent` (the participant's own recent query, `rq_*`) or `generated` (a query written from the participant's memories, `gen_*`, wave 1 only); `query_type` (for generated queries: whether it targets memories that are relevant and useful, or only topically related) |
| `memory_condition` | `with_memory` (all nine retrieved memories) or `filtered_memory` (only the memories the participant marked relevant, listed in `filtered_memory_ids`) |
| `memories` | the nine retrieved memories: `id`, `type` (`persona` = profile statement, `event` = recent event), `text`, `similarity`; plus `redacted: true` or `edited: true` where de-identification changed the text |
| `response_a`, `response_b` | `condition` and `text`, in the order shown to the participant |
| `memory_side` | which slot holds the memory-conditioned response |
| `label.preferred_response` | `a_much` or `b_much` |
| `label.memory_preferred` | `true` if the memory-conditioned response was preferred |
| `label.relevant_memory_ids` | memories the participant marked relevant to the query |
| `label.memory_usefulness` | for relevant memories: `essential`, `clearly_useful` or `somewhat_useful` |
| `label.memory_not_useful_reasons` | for the other memories: `irrelevant`, `relevant_not_useful`, `outdated`, `privacy_intrusive` or `other` (with `other_text`) |
| `label.notes` | optional free-text explanation |
| `label.time_spent_seconds` | time on the item |
| `automatic_judges` | outputs of the screening judges used to select pairs for annotation (Section 3.2): `judge_score`/`judge_reason` (a 1–10 difference rating, available for some wave-1 pairs) and `causal_judge` (wave 2); prompts in `../prompts/screening/` |

The A/B order, labels, conditions and generators are exactly as recorded. Placeholders such as
`[PERSON_1]` or `[LOCATION_2]` replace the same entity everywhere in one participant's records,
including inside the model responses.

## De-identification

- Recruitment IDs, platform IDs, names, timestamps, timings, batch labels and attention-check
  details are removed. Participant IDs are random.
- Only the nine memories retrieved for each pair are released, not the full profile, the full
  event list or the other queries.
- Contact details are masked. Exact ages are replaced by ranges; towns, districts, local
  venues, organisations, events and personal names are replaced by typed, numbered placeholders.
- Every memory, query, note and screening rationale was reviewed for special-category data
  (GDPR Art. 9: health, religion, ethnic origin, political opinion, sex life or orientation),
  offence data (Art. 10) and information about third parties. Counted per pair, 23 memory
  texts were withheld (`redacted: true`) and 33 edited (`edited: true`, including two exact ages).
- Responses were not rewritten, because the participant judged the original text. Where a
  response relied on withheld information, the whole pair was withheld (27 pairs, including
  two participants excluded entirely). Four kept responses have a single-term substitution.

Removing information from free text reduces disclosure risk but cannot guarantee anonymity.

## Intended use and limitations

Intended for research on how assistants should use personal memory: analysing user
preferences, and evaluating or training preference predictors and reward models.

- Small: 262 pairs from 73 participants. Report participant-clustered uncertainty.
- Pairs were screened for substantial, memory-grounded differences (Section 3.2) and only clear
  preferences are included, so win rates are not population estimates.
- Withheld pairs concentrate on health, religion and family topics, so the release
  under-represents personalisation on sensitive topics.
- Generators are confounded with collection wave and participants (Claude-Opus-5 responses come
  from 9 wave-2 participants).

Please do not attempt to identify participants or link records to other data, and do not use
the data to build profiles of individuals. If you find content that you believe identifies a
person, report it to the contact in the main README and we will remove it.

## License

CC BY-NC 4.0 (see `LICENSE`).
