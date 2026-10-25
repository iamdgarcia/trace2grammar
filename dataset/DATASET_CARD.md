# TraceGrammar-Anchor

Reference-outcome dataset accompanying "What Are We Not Testing? Region Coverage of
Conversation Trajectories from Agent Observability Traces". Three third-party corpora,
one flat schema per corpus, every episode carrying (a) a reference outcome produced by
someone OTHER than the walker and (b) the walker's verdict on the transcript alone, so
the two can be confronted without touching either.

Released 2026-10-25. Built by scripts in the artifact repo (`sgd_tau.py`, `tb_tau.py`,
`multiwoz_tau.py` in runtime/); this directory is data + card + checker only.

## Corpora and sizes

| corpus | rows | reference outcome ("truth") | walker columns |
|---|---|---|---|
| taubench/episodes.csv | 1,980 | `reward` (0/1, verified by the tau-bench environment's write checks) | `outcome_convo` (terminal stripped), D, error_count, n_states, reached_close, silent_failure_candidate |
| sgd/dialogs.csv | 4,201 | `gt_effective_call` (true iff at least one service_call in the dialog returned non-empty results) | `outcome_verbatim`, `outcome_calibrated` (dev-calibrated close vocabulary), `verdict_changed`, D (both), `silent_failure_candidate` (recomputed per dialog) |
| multiwoz/dialogs.csv | 1,000 | none (dataset-native domain labels only; no outcome truth exists in MultiWOZ) | `outcome` (terminal stripped), D, n_states, reached_close |

## Provenance (all third-party, fetched verbatim; no relabeling)

- **tau-bench**: official `historical_trajectories` eval runs of sierra-research/tau-bench
  (gpt-4o: 660; claude-3.5-sonnet: 1,320; airline + retail), fetched from
  raw.githubusercontent.com on 2026-10-25 (`runtime/taubench_anchor/tb_to_events.py`).
- **SGD**: dstc8-schema-guided-dialogue official TEST split (dialogues_001..034.json,
  4,201 dialogs, 20 services), fetched 2026-10-25 (`runtime/sgd_validation/`).
- **MultiWOZ 2.2**: test split, first 1,000 dialogs via the HF datasets-server mirror
  `tuetschek/multi_woz_v22` (config default), fetched 2026-10-23
  (`runtime/multiwoz_fetch.py`).

## Label semantics

- tau-bench `reward`: the environment replays every database write; 1.0 only if all task
  checks pass. This is DELIVERED outcome, independent of what the agent said.
- SGD `gt_effective_call`: STRUCTURAL truth — the simulated API actually returned data
  for at least one service call. Delivered-result availability, not user satisfaction.
- MultiWOZ: domain labels are dataset-native; there is NO outcome truth (a dialog has no
  recorded success/failure), so verdicts here cannot be scored against reward — the
  corpus validates transfer, not agreement.
- walker `outcome_*`: theta_cont / theta_aban / theta_hand from
  `trace2grammar.extract.infer_states` with the terminal event stripped (verdict on the
  dialogue alone). theta_esc is never emitted by the walker (escape is a post-hoc audit
  check, per the paper).
- SGD `outcome_calibrated`: same walker, close vocabulary extended with patterns selected
  on the SGD DEV split only (never the test split; published agreement 83.19% vs 55.34%
  verbatim). `verdict_changed=True` marks the 1,172 dialogs the calibration flips — ALL
  in the abandoned->contained direction (added close vocabulary matches closes the
  verbatim tables missed). The 5 courtesy-without-solution dialogs (contained with
  `n_effective==0`) stay contained under both configs.

## KNOWN ARTIFACTS (read before scoring anything)

1. **tau-bench harness cutout**: 82.7% of PASSING episodes carry the walker's abandoned
   shape because the official harness ends episodes with a user `###STOP###` token
   instead of a conversational close. A benchmark that ends episodes by truncation
   cannot exercise abandonment; the shape distribution here measures the harness.
2. **outcome_shipped excluded**: the shipped tau-bench dialect injects the terminal from
   reward by design, so that column is circular and is NOT shipped.
3. **D == tool-retry depth in tau-bench**: D>0 coincides with >=1 failed tool call in
   1,980/1,980 episodes (unlike the human corpora, where D counts clarify re-opens).
4. **SGD silent-failure flag semantics**: `silent_failure_candidate` ships at the
   EXTRACTION level (tool ok + empty result set), which is the published signal
   (107 contained + 91 abandoned dialogs, reproduced exactly for this release). The
   infer_states-level variant of the flag has different semantics (198 dialogs) and is
   intentionally NOT shipped. The 5 courtesy-without-solution dialogs (contained &
   `n_effective==0`) are the strict confirmed subset.
5. **MultiWOZ domain labels are multi-service**: `services` is pipe-joined;
   `primary_service` is the label used in the paper's close-rate table.

## Checksums

See `MANIFEST.sha256` (sha256 of every released file, generated after the last write).

## Licenses (fetched 2026-10-25, findings verbatim)

- tau-bench (sierra-research/tau-bench, LICENSE file): `MIT License`
- SGD (google-research-datasets/dstc8-schema-guided-dialogue): `FETCH FAILED (main+master, no LICENSE file at repo root). GitHub repo license metadata (fetched 2026-10-25): {'key': 'cc-by-sa-4.0', 'name': 'Creative Commons Attribution Share Alike 4.0 International', 'spdx_id': 'CC-BY-SA-4.0', 'url': 'https://api.github.com/licenses/cc-by-sa-4.0', 'node_id': 'MDc6TGljZW5zZTI2'}`
- MultiWOZ: **AMBIGUOUS, documented as found** — code repo budzianowski/multiwoz ships
  `The MIT License (MIT)` (LICENSE file) and its README states MIT;
  the HF mirror we distributed from (`tuetschek/multi_woz_v22`) declares
  `tuetschek/multi_woz_v22 cardData license: apache-2.0`; the MultiWOZ 2.2 paper lineage is commonly
  cited as CC BY-SA 4.0. Nothing in this release re-distributes the raw corpus: we ship
  ids, labels and walker verdicts derived from the transcripts.

Full fetched text: `_license_findings.json`.
