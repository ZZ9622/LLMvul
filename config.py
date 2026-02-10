"""Central config. Override via env (LLMVUL_MODEL, LLMVUL_OUTPUT_DIR)."""
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_MODEL_NAME = "Chun9622/llmvul-finetuned-gemma"
MODEL_NAME = os.environ.get("LLMVUL_MODEL", DEFAULT_MODEL_NAME)

DATA_DIR = os.path.join(ROOT_DIR, "data")
VUL_JSONL = os.path.join(DATA_DIR, "primevul236.jsonl")
NONVUL_JSONL = os.path.join(DATA_DIR, "primenonvul236.jsonl")
TP_TN_SAMPLES_JSONL = os.path.join(DATA_DIR, "tp_tn_samples.jsonl")
NEURON_ANALYSIS_JSON = os.path.join(DATA_DIR, "neuron_analysis.json")

OUTPUT_DIR = os.environ.get("LLMVUL_OUTPUT_DIR", "")
LOG_DIR = os.path.join(OUTPUT_DIR, "log") if OUTPUT_DIR else ""
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots") if OUTPUT_DIR else ""


def get_device():
    try:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


DEVICE = get_device()


def _circuit_tracer_path():
    p = os.path.join(ROOT_DIR, "circuit-tracer", "circuit-tracer")
    if os.path.isdir(p) and os.path.isfile(os.path.join(p, "circuit_tracer", "__init__.py")):
        return p
    return None


CIRCUIT_TRACER_PATH = _circuit_tracer_path()


def setup_circuit_tracer():
    """Add circuit-tracer to sys.path if not importable."""
    import sys
    try:
        import circuit_tracer  # noqa: F401
        return
    except ImportError:
        pass
    if CIRCUIT_TRACER_PATH and CIRCUIT_TRACER_PATH not in sys.path:
        sys.path.insert(0, CIRCUIT_TRACER_PATH)
