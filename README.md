# LLMvul

LLM-based code vulnerability detection and interpretability: main pipeline (Prime) and downstream analyses (attention, causal patching, causal validation, circuit visualization, statistical analysis).

## Model

Fine-tuned Gemma for vulnerability detection:

- **HuggingFace**: [Chun9622/llmvul-finetuned-gemma](https://huggingface.co/Chun9622/llmvul-finetuned-gemma)

## Requirements

- Python ≥ 3.10
- CUDA (optional, for GPU)

## Installation

   ```bash
   git clone https://github.com/ZZ9622/LLMvul.git
   cd LLMvul
   # Optional: for circuit tracing
   git clone https://github.com/decoderesearch/circuit-tracer.git
   pip install -e circuit-tracer/circuit-tracer
   pip install -r requirements.txt
   ```

## Data

Please see data/README.md

### Demo 

Quick run on the first 5 samples from the HuggingFace dataset:

```bash
python demo/demo.py
```

Output appears under `demo/output/`. Model and dataset are downloaded from HuggingFace on first run.

### HPC (run all scripts)

On a Slurm cluster, from repo root:

```bash
sbatch demo/run_all.sbatch
```

## License

See repository license file.
