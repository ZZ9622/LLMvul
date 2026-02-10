# LLMvul

LLM-based code vulnerability detection and interpretability: main pipeline (Prime) and downstream analyses (attention, causal patching, causal validation, circuit visualization, statistical analysis).

## Model

Fine-tuned Gemma for vulnerability detection:

- **HuggingFace**: [Chun9622/llmvul-finetuned-gemma](https://huggingface.co/Chun9622/llmvul-finetuned-gemma)

## Requirements

- Python ≥ 3.10
- CUDA (optional, for GPU)

## Installation

1. Clone the repo and optionally circuit-tracer:

   ```bash
   git clone https://github.com/ZZ9622/LLMvul.git
   cd LLMvul
   # Optional: for attribution / circuit tracing
   git clone https://github.com/neelsald/circuit-tracer.git
   pip install -e circuit-tracer/circuit-tracer
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Put data under `data/` (see [Data](#data)).

## Data

Under `data/`:

- `primevul236.jsonl` – vulnerable samples (JSONL: `func`, `target`, `idx`, etc.)
- `primenonvul236.jsonl` – non-vulnerable samples
- `tp_tn_samples.jsonl` – TP/TN samples (attention & causal scripts)
- `neuron_analysis.json` – top neuron IDs per layer (causal validation)

Schema: `data/README.md`.

## Usage

Run from repo root. Optional: `export LLMVUL_OUTPUT_DIR=./out` for logs/plots; if unset, no output dirs are created.

### Main entry (run by keyword)

```bash
python scripts/main.py <keyword> [extra args...]
```

| Keyword | Description |
|---------|-------------|
| `prime` | Main pipeline: prediction + L0 attribution |
| `attention` | Attention head importance |
| `causal_patching` | Causal patching experiment |
| `causal_validation` | Causal validation / ablation |
| `advanced_statistical` | Advanced stats (e.g. L2 norms) |
| `analyze_circuits` | Circuit analysis (pass prime output JSON path) |
| `circuitplot` | Circuit attribution & visualization |

Examples:

```bash
python scripts/main.py prime
python scripts/main.py attention
python scripts/main.py analyze_circuits ./out/log/20250101_120000/out.json
python scripts/main.py circuitplot
```

You can also run scripts directly, e.g. `python scripts/prime.py`. Env vars: `PRED_BATCH_SIZE`, `MAX_NEW_TOKENS`, `MAX_INPUT_TOKENS`, `LOG_EVERY`.

## Project layout

```
LLMvul/
├── config.py           # Paths, model (override via env)
├── requirements.txt
├── data/
├── scripts/
│   ├── main.py         # Entry: dispatch by keyword
│   ├── prime.py
│   ├── attention_analysis.py
│   ├── causal_patching.py
│   ├── causal_validation.py
│   ├── advanced_statistical.py
│   ├── analyze_circuits.py
│   ├── circuitplot.py
│   └── visualize_custom.py
└── circuit-tracer/     # Optional clone
```

## License

See repository license file.
