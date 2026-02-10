# Data

Place the following files in this directory.

## Files

- **primevul236.jsonl** – Vulnerable code samples (one JSON object per line).
- **primenonvul236.jsonl** – Non-vulnerable code samples.
- **tp_tn_samples.jsonl** – True positive / true negative samples used by attention and causal scripts. Each line is a JSON object with at least: `func`, `prediction_type` (e.g. `"TP"` / `"TN"`), `cwe` (list).
- **neuron_analysis.json** – Per-layer top neuron IDs for causal validation. Format: `{ "layer_idx": [ {"neuron_idx": int, ...}, ... ], ... }`.

## JSONL schema (primevul / primenonvul)

Each line is a JSON object with at least:

- `func`: string (code snippet)
- `target`: 0 (non-vulnerable) or 1 (vulnerable)
- `idx`: optional sample id
- `cwe`: optional list of CWE ids

Other fields (e.g. `project`, `commit_id`, `cve`) may be present.
