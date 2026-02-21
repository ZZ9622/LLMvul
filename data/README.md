# data/

Place data files here **or** let scripts download them automatically from
HuggingFace ([Chun9622/LLMvul](https://huggingface.co/datasets/Chun9622/LLMvul)).

## Expected files

| File | Description | Produced by |
|------|-------------|-------------|
| `primevul236.jsonl` | Vulnerable code samples | HF auto-download |
| `primenonvul236.jsonl` | Non-vulnerable code samples | HF auto-download |
| `tp_tn_samples.jsonl` | TP/TN predictions from prime.py | `scripts/prime.py` |
| `neuron_analysis.json` | Top neuron IDs per layer | (optional, for causal validation) |

## JSONL schema

Each line in `primevul236.jsonl` / `primenonvul236.jsonl`:

```json
{
  "idx":    196316,
  "func":   "void foo(...) { ... }",
  "target": 1,
  "cwe":    ["CWE-119"]
}
```

Each line in `tp_tn_samples.jsonl`:

```json
{
  "idx":             196316,
  "func":            "void foo(...) { ... }",
  "cwe":             ["CWE-119"],
  "prediction_type": "TP",
  "pred_label":      "vul",
  "model_output":    "Vulnerable …"
}
```
