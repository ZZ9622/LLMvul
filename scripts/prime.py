#!/usr/bin/env python3
import os
import sys
# ── Repository root & output directory (auto-detected) ───────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.dirname(_SCRIPT_DIR)           # LLMvul/
OUTPUT_BASE = os.environ.get(
    "LLMVUL_OUTPUT_DIR", os.path.join(ROOT_DIR, "out")
)
# ── circuit-tracer: prefer installed package, fall back to local clone ────────
_CT_PATH = os.path.join(ROOT_DIR, "circuit-tracer", "circuit-tracer")
if _CT_PATH not in sys.path:
    sys.path.insert(0, _CT_PATH)
import json
import re
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from transformers import AutoTokenizer
from circuit_tracer.replacement_model import ReplacementModel
from circuit_tracer.attribution.attribute import attribute
import warnings
import gc
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
from contextlib import nullcontext

# === Time stamp & directories ===
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = os.path.join(OUTPUT_BASE, "log", ts)
PLOT_DIR = os.path.join(OUTPUT_BASE, "plots", ts)
predictions_json_path = os.path.join(LOG_DIR, "all_predictions.json")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# === Paths & HuggingFace fallback ===
MODEL_PATH = "Chun9622/llmvul-finetuned-gemma"
DATASET_HF_ID = "Chun9622/LLMvul"

def _ensure_data_jsonl():
    """Return (vul_path, nonvul_path), searching local dirs then downloading from HF."""
    import json as _json
    _data_dir = os.path.join(ROOT_DIR, "data")
    _vul   = os.path.join(_data_dir, "primevul236.jsonl")
    _nonvul = os.path.join(_data_dir, "primenonvul236.jsonl")
    if os.path.exists(_vul) and os.path.exists(_nonvul):
        return _vul, _nonvul

    # Fallback: look in sibling data/ directory (original workspace layout)
    _sibling = os.path.join(os.path.dirname(ROOT_DIR), "data")
    _svul   = os.path.join(_sibling, "primevul236.jsonl")
    _snonvul = os.path.join(_sibling, "primenonvul236.jsonl")
    if os.path.exists(_svul) and os.path.exists(_snonvul):
        print(f"[INFO] Using data files from sibling dir: {_sibling}")
        os.makedirs(_data_dir, exist_ok=True)
        import shutil as _sh
        _sh.copy2(_svul,   _vul)
        _sh.copy2(_snonvul, _nonvul)
        return _vul, _nonvul

    # Last resort: download per-file from HuggingFace (avoids schema-mismatch error)
    print("[INFO] Local data files not found – downloading from HuggingFace …")
    from datasets import load_dataset as _lds  # type: ignore
    os.makedirs(_data_dir, exist_ok=True)
    try:
        ds_vul   = _lds("json",
                        data_files=f"hf://datasets/{DATASET_HF_ID}/primevul236.jsonl",
                        split="train")
        ds_nonvul = _lds("json",
                         data_files=f"hf://datasets/{DATASET_HF_ID}/primenonvul236.jsonl",
                         split="train")
        vul_recs, nonvul_recs = list(ds_vul), list(ds_nonvul)
    except Exception as _e:
        print(f"[WARN] Per-file HF download failed ({_e}); retrying with full dataset …")
        ds = _lds(DATASET_HF_ID, ignore_verifications=True)
        _split = ds.get("train") or ds[list(ds.keys())[0]]
        vul_recs   = [r for r in _split if r.get("target") == 1]
        nonvul_recs = [r for r in _split if r.get("target") == 0]
    for recs, fpath in [(vul_recs, _vul), (nonvul_recs, _nonvul)]:
        with open(fpath, "w", encoding="utf-8") as _f:
            for r in recs:
                _f.write(_json.dumps(dict(r), ensure_ascii=False) + "\n")
    print(f"[INFO] Data saved to {_data_dir}")
    return _vul, _nonvul

VUL_PATH, NONVUL_PATH = _ensure_data_jsonl()
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
MAX_FEATURE_NODES = 200
ATTR_MAX_WORKERS = 1  # keep 1 to avoid attribution conflicts
PREFETCH = 12         # increase prefetch queue depth
PRED_BATCH_SIZE = int(os.getenv("PRED_BATCH_SIZE", "8"))   # prediction batch size (overridable via env)
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "200"))   # generation length (increased to 200 for complete answers)
MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", "512"))  # input truncation length (default 512)
LOG_EVERY = int(os.getenv("LOG_EVERY", "10"))  # log every N samples

# === Logger ===
class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()
sys.stdout = Tee(sys.stdout, open(os.path.join(LOG_DIR, f"log_{ts}.txt"), "w"))
sys.stderr = sys.stdout

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
warnings.filterwarnings('ignore')

# Enable Rust tokenizer parallelism
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

# Read-write lock allows concurrent reads
class ReadWriteLock:
    def __init__(self):
        self._read_ready = threading.Condition(threading.RLock())
        self._readers = 0
        self._writers = 0
        self._write_waiters = 0
    
    def acquire_read(self):
        self._read_ready.acquire()
        try:
            while self._writers > 0 or self._write_waiters > 0:
                self._read_ready.wait()
            self._readers += 1
        finally:
            self._read_ready.release()
    
    def release_read(self):
        self._read_ready.acquire()
        try:
            self._readers -= 1
            if self._readers == 0:
                self._read_ready.notify_all()
        finally:
            self._read_ready.release()
    
    def acquire_write(self):
        self._read_ready.acquire()
        self._write_waiters += 1
        try:
            while self._readers > 0 or self._writers > 0:
                self._read_ready.wait()
            self._write_waiters -= 1
            self._writers += 1
        finally:
            self._read_ready.release()
    
    def release_write(self):
        self._read_ready.acquire()
        try:
            self._writers -= 1
            self._read_ready.notify_all()
        finally:
            self._read_ready.release()

model_lock = ReadWriteLock()

# === Patch for model loading ===
def patch_model_loading():
    import transformer_lens.loading_from_pretrained as loading
    original = loading.get_official_model_name
    loading.get_official_model_name = (
        lambda model_name: "google/gemma-2-2b"
        if model_name == MODEL_PATH
        else original(model_name)
    )
def patch_model_config_loading():
    import transformer_lens.loading_from_pretrained as loading
    original = loading.get_pretrained_model_config
    def patched(model_name, **kwargs):
        if model_name == MODEL_PATH:
            from transformers import AutoConfig
            return AutoConfig.from_pretrained(model_name)
        return original(model_name, **kwargs)
    loading.get_pretrained_model_config = patched
patch_model_loading()
patch_model_config_loading()

# === Load model & tokenizer ===
print("[INFO] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("[INFO] Tokenizer loaded.")

# 允许 TF32（Ampere+），并尝试提高 matmul 精度策略以提速
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

print("[INFO] Loading ReplacementModel...")
rm = ReplacementModel.from_pretrained(
    MODEL_PATH,
    transcoder_set="gemma",
    device=DEVICE,
    torch_dtype=torch.float16 if DEVICE.startswith("cuda") else torch.float32
)
print("[INFO] ReplacementModel loaded.")

# === Set eval ===
rm.eval()
print("[INFO] Model set to eval mode.")

# Create CUDA streams for concurrent operations
if torch.cuda.is_available():
    pred_stream = torch.cuda.Stream()
    attr_streams = [torch.cuda.Stream() for _ in range(ATTR_MAX_WORKERS)]
else:
    pred_stream = None
    attr_streams = [None] * ATTR_MAX_WORKERS

# === Utilities ===
def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def safe_tensor_to_python(x):
    if isinstance(x, torch.Tensor):
        return x.item() if x.numel() == 1 else x.detach().cpu().numpy().tolist()
    return x

def extract_feature_info(feature):
    info = {}
    try:
        vals = [safe_tensor_to_python(feature[i]) for i in range(min(4, len(feature)))]
        if len(vals) >= 2:
            info['layer'] = int(vals[0])
            info['position'] = int(vals[1])
    except:
        pass
    return info

def compute_l0_per_layer(graph):
    l0_per_layer = {}
    if hasattr(graph, 'active_features'):
        for f in graph.active_features:
            info = extract_feature_info(f)
            layer = info.get('layer', None)
            if isinstance(layer, int):
                l0_per_layer[layer] = l0_per_layer.get(layer, 0) + 1
    return l0_per_layer

def extract_label(text):
    """
    Extract vul/nonvul label from model output.
    
    v6 (2025-10-25): Enhanced to better handle code continuation and truncated outputs
    Problem: 67% unknown due to code continuation being truncated before reaching answer
    Solution: More aggressive pattern matching and fallback strategies
    
    Priority strategy:
    0. Find "Answer:" and extract text after it
    1. Look for "safe" or "vulnerable" keywords anywhere in output
    2. Check for code-followed-by-answer pattern
    3. Detect question echo patterns
    4. Keyword analysis with lower threshold
    """
    import re
    
    if not text or len(text.strip()) == 0:
        return "unknown"
    
    t = text.lower().strip()
    original_text = text.strip()
    
    # === PRIORITY 0: Extract text AFTER "Answer:" marker ===
    answer_match = re.search(r'answer:\s*(.+)', t, re.IGNORECASE | re.DOTALL)
    if answer_match:
        answer_text = answer_match.group(1).strip()
        if answer_text:
            answer_lines = answer_text.split('\n')
            answer_first_line = answer_lines[0].strip()
            answer_words = answer_first_line.split()[:15]  # Increased from 10 to 15
            
            if len(answer_words) > 0:
                first_word = answer_words[0]
                
                # Check for direct vulnerability statement
                if 'vulnerable' in answer_first_line[:100] or 'unsafe' in answer_first_line[:100]:
                    if 'not vulnerable' not in answer_first_line[:100] and 'not unsafe' not in answer_first_line[:100]:
                        return "vul"
                
                # Check for safety statement
                if 'safe' in answer_first_line[:100] or 'secure' in answer_first_line[:100]:
                    if 'not safe' not in answer_first_line[:100] and 'unsafe' not in answer_first_line[:100]:
                        return "nonvul"
                
                # Check first word patterns
                if first_word in ['it', 'this', 'the']:
                    if 'vulnerable' in answer_first_line[:100]:
                        return "vul"
                    if 'safe' in answer_first_line[:100] or 'secure' in answer_first_line[:100]:
                        return "nonvul"
                
                # Direct answer words
                if first_word in ['vulnerable', 'unsafe', 'insecure', 'yes']:
                    if first_word == 'yes':
                        if len(answer_words) >= 2 and 'vulnerable' in answer_first_line[:60]:
                            return "vul"
                    else:
                        return "vul"
                
                if first_word in ['safe', 'secure', 'no']:
                    if first_word == 'no':
                        return "nonvul"
                    return "nonvul"
    
    # === PRIORITY 1: Search for "Question:" followed by answer ===
    # Model might echo question, then answer
    question_match = re.search(r'question:.*?answer:\s*(.+)', t, re.IGNORECASE | re.DOTALL)
    if question_match:
        answer_part = question_match.group(1).strip()
        if 'vulnerable' in answer_part[:50] and 'not vulnerable' not in answer_part[:50]:
            return "vul"
        if 'safe' in answer_part[:50] and 'not safe' not in answer_part[:50]:
            return "nonvul"
    
    # === PRIORITY 2: Look for answer keywords ANYWHERE in output ===
    # Model might have given answer after code continuation
    full_text = t[:800]  # Increased search window from 400 to 800
    
    # Strong vulnerability indicators
    if re.search(r'\b(is|be|appears?|seems?)\s+(vulnerable|unsafe|insecure)', full_text):
        if 'not vulnerable' not in full_text and 'not unsafe' not in full_text:
            return "vul"
    
    if re.search(r'\bvulnerable\s+to\b', full_text):
        return "vul"
    
    if re.search(r'\b(this|the)\s+(code|function|program)\s+(is|has|contains)\s+(a\s+)?(vulnerable|vulnerability|unsafe|flaw|bug)', full_text):
        return "vul"
    
    # Check for specific vulnerability types
    if re.search(r'\b(buffer\s+overflow|memory\s+leak|injection|overflow|underflow|race\s+condition)', full_text):
        # If vulnerability type mentioned, likely vulnerable
        if re.search(r'\b(fix|prevent|avoid|safe\s+from)\s+', full_text):
            # But if in context of prevention, might be safe
            pass
        else:
            return "vul"
    
    # Strong safety indicators
    if re.search(r'\b(is|be|appears?|seems?)\s+(safe|secure|correct)', full_text):
        if 'not safe' not in full_text and 'unsafe' not in full_text and 'not secure' not in full_text:
            return "nonvul"
    
    if re.search(r'\b(this|the)\s+(code|function|program)\s+(is|appears)\s+(safe|secure|correct)', full_text):
        return "nonvul"
    
    # "no vulnerabilities" or "no issues"
    if re.search(r'\bno\s+(vulnerabilities|security\s+issues|flaws|bugs|problems)', full_text):
        return "nonvul"
    
    # === PRIORITY 3: Check first meaningful line ===
    lines = t.split('\n')
    for line in lines[:5]:  # Check first 5 lines instead of just first
        line = line.strip()
        if len(line) < 5:  # Skip very short lines
            continue
        words = line.split()[:5]
        if len(words) > 0:
            first_word = words[0]
            # Skip code-like patterns
            if first_word in ['const', 'void', 'int', 'static', 'struct', 'char', 'return', 'typedef', 
                            'if', 'for', 'while', 'switch', 'case', '{', '}', '//', '/*']:
                continue
            # Check for direct answer
            if first_word in ['vulnerable', 'unsafe', 'insecure']:
                return "vul"
            if first_word in ['safe', 'secure']:
                return "nonvul"
            # Found a meaningful line, analyze it
            if 'vulnerable' in line or 'unsafe' in line:
                return "vul"
            if 'safe' in line and 'unsafe' not in line:
                return "nonvul"
    
    # === PRIORITY 4: Keyword counting with LOWER threshold ===
    # Be more aggressive to reduce unknowns
    first_500 = t[:500]  # Increased from 300
    
    safe_keywords = ['safe', 'secure', 'correct', 'valid', 'okay', 'fine', 'proper', 'protected', 'good']
    vul_keywords = ['vulnerable', 'unsafe', 'insecure', 'flaw', 'bug', 'exploit', 
                    'overflow', 'leak', 'injection', 'attack', 'danger', 'malicious', 'risk']
    
    safe_count = sum(first_500.count(kw) for kw in safe_keywords)
    vul_count = sum(first_500.count(kw) for kw in vul_keywords)
    
    # LOWERED threshold: even 1 occurrence with no opposition
    if vul_count >= 1 and safe_count == 0:
        return "vul"
    
    if safe_count >= 1 and vul_count == 0:
        return "nonvul"
    
    # If both present, use ratio with lower threshold
    if vul_count > safe_count and vul_count >= 1:
        if vul_count >= safe_count * 1.5:  # Lowered from 2x to 1.5x
            return "vul"
    
    if safe_count > vul_count and safe_count >= 1:
        if safe_count >= vul_count * 1.5:  # Lowered from 2x to 1.5x
            return "nonvul"
    
    # === PRIORITY 5: Last resort - check for ANY security-related content ===
    # If output contains security discussion but we can't determine direction, 
    # check sentiment
    security_words = ['security', 'vulnerability', 'safe', 'unsafe', 'secure', 'insecure']
    has_security_content = any(word in first_500 for word in security_words)
    
    if has_security_content:
        # Count positive vs negative sentiment
        positive = first_500.count('safe') + first_500.count('secure') + first_500.count('correct')
        negative = first_500.count('vulnerable') + first_500.count('unsafe') + first_500.count('flaw')
        
        if positive > negative:
            return "nonvul"
        elif negative > positive:
            return "vul"
    
    # === If we get here, output is truly ambiguous ===
    return "unknown"
# ============================================================================
# Attribution and Analysis Functions
# ============================================================================

def run_prediction_batch(samples):
    """
    Optimized batched prediction using a dedicated CUDA stream.
    """
    if not samples:
        return []
    
    prompts = [s["prompt"].strip() if isinstance(s, dict) else s.strip() for s in samples]
    
    # Use dedicated stream for prediction
    stream_context = torch.cuda.stream(pred_stream) if pred_stream else nullcontext()
    
    with stream_context:
    # Limit input length and enable pinned memory to accelerate H2D
        enc = tokenizer(prompts, return_tensors="pt", truncation=True, max_length=MAX_INPUT_TOKENS, padding=True)
    # Pin tensors in host memory before moving to device
        enc = {k: (v.pin_memory() if isinstance(v, torch.Tensor) else v) for k, v in enc.items()}
        input_ids = enc["input_ids"].to(DEVICE, non_blocking=True)
        attn = enc.get("attention_mask")
        
        if attn is not None:
            attn = attn.to(DEVICE, non_blocking=True)
            prompt_lens = attn.sum(dim=1).cpu().tolist()
        else:
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            prompt_lens = [(row != pad_id).sum().item() for row in enc["input_ids"]]

        # inference_mode has lower overhead than no_grad
        with torch.inference_mode():
            # Acquire read lock to allow concurrent inference
            model_lock.acquire_read()
            try:
                output_ids = rm.generate(
                    input_ids,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    verbose=False  # disable progress bar
                )
            finally:
                model_lock.release_read()
        
    # Decode only the generated portion
        gen_slices = []
        for i in range(len(samples)):
            pl = int(prompt_lens[i])
            gen_slices.append(output_ids[i][pl:])  # Only generated tokens
        
        decoded_gen = tokenizer.batch_decode(gen_slices, skip_special_tokens=True)

        results = []
        for i in range(len(samples)):
            pl = int(prompt_lens[i])
            # Use generated-only text for prediction
            out_text = decoded_gen[i].strip()
            
            # If generated output is empty, something went wrong
            if not out_text:
                print(f"[WARN] Sample {i}: Empty generation.")
                out_text = decoded_gen[i]  # Keep even if empty
            
            pred = extract_label(out_text)
            # Store only generated text
            results.append((pred, out_text, prompts[i], pl))
    
    # Synchronize stream to ensure data is ready
    if pred_stream:
        pred_stream.synchronize()
    
    # Delayed cleanup to reduce frequency
    del enc, input_ids, output_ids
    
    return results

def attribute_task(prompt, worker_id=0, max_nodes_init=MAX_FEATURE_NODES):
    """
    Optimized attribution task with better error handling.
    """
    max_nodes = max_nodes_init
    # Note: Avoid CUDA streams for attribution due to potential internal state conflicts
    # stream = attr_streams[worker_id % len(attr_streams)] if attr_streams[0] else None
    # stream_context = torch.cuda.stream(stream) if stream else nullcontext()
    
    for attempt in range(3):
        try:
            # Clean old states before retry
            if attempt > 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Acquire write lock to ensure exclusivity for attribution (avoid activation conflicts)
            with torch.enable_grad():
                model_lock.acquire_write()
                try:
                    # Ensure model is in correct state
                    rm.eval()
                    
                    g = attribute(
                        prompt=prompt,
                        model=rm,
                        max_n_logits=3,
                        batch_size=BATCH_SIZE,
                        max_feature_nodes=max_nodes,
                        verbose=False
                    )
                finally:
                    model_lock.release_write()
            
            l0 = compute_l0_per_layer(g)
            del g
            
            # Reduce cleanup frequency
            if attempt > 0 or worker_id % 3 == 0:
                gc.collect()
            
            return l0
            
        except RuntimeError as e:
            error_str = str(e).lower()
            if "cuda out of memory" in error_str or "out of memory" in error_str:
                print(f"[OOM] Attribution attempt {attempt+1} failed (max_nodes={max_nodes}). Reducing parameters...")
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                max_nodes = max(30, int(max_nodes * 0.6))
                continue
            elif "weak reference" in error_str or "weakproxy" in error_str:
                print(f"[WARN] Attribution attempt {attempt+1} failed due to weak reference issue. Retrying...")
                gc.collect()
                max_nodes = max(50, int(max_nodes * 0.8))
                continue
            elif "size of tensor" in error_str or "must match" in error_str:
                # Tensor size mismatch, possibly due to sequence length
                print(f"[WARN] Tensor size mismatch (attempt {attempt+1}). Retrying with reduced nodes...")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                max_nodes = max(40, int(max_nodes * 0.7))
                continue
            else:
                print(f"[ERROR] Attribution RuntimeError: {e}")
                if attempt < 2:
                    gc.collect()
                    max_nodes = max(50, int(max_nodes * 0.75))
                    continue
                return {}
        except KeyError as e:
            # 'acts' key error usually indicates activations not stored correctly
            print(f"[ERROR] Attribution KeyError (attempt {attempt+1}): {e}")
            if attempt < 2:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # 尝试更保守的参数
                max_nodes = max(30, int(max_nodes * 0.5))
                continue
            else:
                return {}
        except (TypeError, ValueError, AttributeError) as e:
            print(f"[ERROR] Attribution type/value error (attempt {attempt+1}): {e}")
            if attempt < 2:
                gc.collect()
                max_nodes = max(50, int(max_nodes * 0.7))
                continue
            else:
                return {}
        except Exception as e:
            print(f"[ERROR] Attribution unexpected error (attempt {attempt+1}): {type(e).__name__}: {e}")
            if attempt < 2:
                gc.collect()
                continue
            else:
                return {}
    
    print(f"[FAILED] Attribution failed after all retries (final max_nodes={max_nodes})")
    return {}

def process_samples_with_attr_pool(samples, tag):
    """
    Optimized pipeline with better concurrency control.
    """
    # results: only samples with successful L0 attribution
    results = []
    # predictions_all: predictions for ALL samples regardless of attribution
    predictions_all = []
    q = queue.Queue(maxsize=PREFETCH)
    total = len(samples)
    stop_token = object()

    def producer():
        idx = 0
        n = len(samples)
        while idx < n:
            batch = samples[idx: idx + PRED_BATCH_SIZE]
            batch_results = run_prediction_batch(batch)
            for j, (pred, output_text, prompt, prompt_len) in enumerate(batch_results):
                k = idx + j + 1
                if k % LOG_EVERY == 0 or k == 1 or k == total:
                    print(f"[{tag}] Sample {k}/{total} [len={prompt_len}]")
                sample = batch[j]
                true_label = sample.get("true_label", "unknown") if isinstance(sample, dict) else "unknown"
                sample_idx = sample.get("idx", idx + j) if isinstance(sample, dict) else (idx + j)
                q.put({
                    "idx": sample_idx,
                    "pred": pred,
                    "true_label": true_label,
                    "output_text": output_text,
                    "prompt": prompt
                })
            idx += len(batch)
            
            # 批次之间小幅清理以保持内存稳定
            if idx % (PRED_BATCH_SIZE * 5) == 0:
                gc.collect()
        
        q.put(stop_token)

    prod_thread = threading.Thread(target=producer, daemon=True)
    prod_thread.start()

    in_flight = {}
    completed = 0
    worker_counter = 0
    
    with ThreadPoolExecutor(max_workers=ATTR_MAX_WORKERS, thread_name_prefix="attr") as executor:
        while True:
            # Submit new tasks until reaching concurrency limit
            while len(in_flight) < ATTR_MAX_WORKERS:
                try:
                    item = q.get(timeout=0.1)
                except queue.Empty:
                    break
                    
                if item is stop_token:
                    q.put(stop_token)
                    break
                
                future = executor.submit(attribute_task, item["prompt"], worker_counter)
                in_flight[future] = item
                worker_counter += 1

            if not in_flight:
                try:
                    tok = q.get(timeout=0.1)
                    if tok is stop_token:
                        break
                    else:
                        q.put(tok)
                        continue
                except queue.Empty:
                    continue

            # Handle completed futures
            done_futures = []
            for fut in list(in_flight.keys()):
                if fut.done():
                    done_futures.append(fut)

            for fut in done_futures:
                meta = in_flight.pop(fut)
                l0 = {}
                try:
                    l0 = fut.result()
                except Exception as e:
                    print(f"[ERROR] Attribute future failed: {e}")
                    l0 = {}

                completed += 1
                idx = meta["idx"]
                pred = meta["pred"]
                true_label = meta["true_label"]
                out = meta["output_text"]
                # Display truncated output in console for readability
                display_out = out[:80] + '...' if len(out) > 80 else out
                print(f"→ Sample {completed}/{total} | True: {true_label} | Pred: {pred} | Output: {display_out}")

                # Record prediction output (generated text only)
                predictions_all.append({
                    "idx": idx,
                    "true_label": true_label,
                    "pred_label": pred,
                    "model_output": out,
                    "sample_type": tag.lower()
                })

                if l0:
                    results.append({
                        "idx": idx,
                        "true_label": true_label,
                        "pred_label": pred,
                        "model_output": out,
                        "l0_per_layer": l0
                    })
                    
                if completed % 50 == 0:
                    success_rate = len(results) / completed * 100
                    print(f"[{tag}] Progress: {completed}/{total} completed, {len(results)} with L0 data ({success_rate:.1f}% success)")
            
            # Avoid busy waiting
            if not done_futures and len(in_flight) >= ATTR_MAX_WORKERS:
                import time
                time.sleep(0.01)

    prod_thread.join()
    
    # Final cleanup
    gc.collect()
    clear_gpu_cache()
    
    return results, predictions_all

def load_prompts(jsonl_path):
    """
    Load prompts from jsonl_path.
    """
    samples = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                code = obj.get("func", "").strip()
                if not code:
                    continue

                target = obj.get("target", -1)
                if target not in [0, 1]:
                    continue

                true_label = "vul" if target == 1 else "nonvul"
                idx = obj.get("idx", -1)

                # Use compact prompt (2025-10-19 v2) - CRITICAL FIX
                # Previous version caused 2 major issues:
                # 1. High FP: Model repeated "B) Safe\n...Answer: A" - we read B first, missed A!
                # 2. Unknown: Model repeated "letter (A or B)" without answering
                # 
                # ROOT CAUSE: Multi-line format with choices gave model too much structure to copy
                # 
                # NEW APPROACH: Compact single-paragraph format
                # CRITICAL FIX v5 (2025-10-20): Simplify to direct question
                # v4 problem: Model continues code, then echoes question, then answers
                # v5 strategy: Very short code snippet + direct question without extra context
                # Put code inline to reduce continuation tendency
                prompt = f"""Code: {code}

Question: Is this code safe or vulnerable?
Answer:"""

                samples.append({
                    "idx": idx,
                    "true_label": true_label,
                    "prompt": prompt
                })

            except Exception as e:
                print(f"[WARN] Failed to parse one line: {e}")
                continue

    return samples

def average_l0_dicts(list_of_dicts):
    from collections import defaultdict
    layer_sums = defaultdict(float)
    count = defaultdict(int)
    for d in list_of_dicts:
        for layer, value in d.items():
            layer_sums[layer] += value
            count[layer] += 1
    avg = {layer: layer_sums[layer]/count[layer] for layer in sorted(layer_sums)}
    return avg

def calculate_classification_metrics(vul_results, nonvul_results):
    """
    Compute classification metrics: TP, FP, TN, FN, accuracy, precision, recall, F1, and unknown counts.
    """
    # Initialize counters
    tp = 0  # True Positive: 预测为vul且真实为vul
    fp = 0  # False Positive: 预测为vul但真实为nonvul
    tn = 0  # True Negative: 预测为nonvul且真实为nonvul
    fn = 0  # False Negative: 预测为nonvul但真实为vul
    
    unknown_vul = 0  # 真实为vul但预测为unknown
    unknown_nonvul = 0  # 真实为nonvul但预测为unknown
    
    # Count for VUL samples (true label == vul)
    for result in vul_results:
        pred = result.get("pred_label", "unknown")
        if pred == "vul":
            tp += 1
        elif pred == "nonvul":
            fn += 1
        else:  # pred == "unknown"
            unknown_vul += 1
    
    # Count for NONVUL samples (true label == nonvul)
    for result in nonvul_results:
        pred = result.get("pred_label", "unknown")
        if pred == "nonvul":
            tn += 1
        elif pred == "vul":
            fp += 1
        else:  # pred == "unknown"
            unknown_nonvul += 1
    
    # Overall counts
    total_samples = len(vul_results) + len(nonvul_results)
    total_unknown = unknown_vul + unknown_nonvul
    total_classified = tp + fp + tn + fn
    
    # Metrics based on non-unknown predictions
    accuracy = (tp + tn) / total_classified if total_classified > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Overall accuracy including unknown
    overall_accuracy = (tp + tn) / total_samples if total_samples > 0 else 0.0
    
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "unknown_vul": unknown_vul,
        "unknown_nonvul": unknown_nonvul,
        "total_unknown": total_unknown,
        "total_samples": total_samples,
        "total_classified": total_classified,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "overall_accuracy": overall_accuracy
    }

def plot_comparison(vul_l0, nonvul_l0, out_path):
    layers = sorted(set(vul_l0) | set(nonvul_l0))
    vul_vals = [vul_l0.get(l, 0) for l in layers]
    nonvul_vals = [nonvul_l0.get(l, 0) for l in layers]
    
    plt.figure(figsize=(12, 6))
    plt.plot(layers, vul_vals, label="VUL", color="red", marker='o')
    plt.plot(layers, nonvul_vals, label="NONVUL", color="blue", marker='s')
    plt.xlabel("Layer")
    plt.ylabel("Average Total L0")
    plt.title("Average Total L₀ per Layer (VUL vs NONVUL)", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[SAVED] Plot saved: {out_path}")

def plot_all_samples_l0_trends(vul_results, nonvul_results, out_path):
    plt.figure(figsize=(15, 8))
    
    for result in vul_results:
        l0_dict = result.get("l0_per_layer", {})
        if l0_dict:
            layers = sorted(l0_dict.keys())
            values = [l0_dict[layer] for layer in layers]
            plt.plot(layers, values, color='red', alpha=0.3, linewidth=0.8)
    
    for result in nonvul_results:
        l0_dict = result.get("l0_per_layer", {})
        if l0_dict:
            layers = sorted(l0_dict.keys())
            values = [l0_dict[layer] for layer in layers]
            plt.plot(layers, values, color='blue', alpha=0.3, linewidth=0.8)
    
    avg_vul = average_l0_dicts([x.get("l0_per_layer", {}) for x in vul_results])
    avg_nonvul = average_l0_dicts([x.get("l0_per_layer", {}) for x in nonvul_results])
    
    if avg_vul:
        vul_layers = sorted(avg_vul.keys())
        vul_vals = [avg_vul[l] for l in vul_layers]
        plt.plot(vul_layers, vul_vals, color='red', linewidth=3, 
                label=f'VUL Average (n={len(vul_results)})', marker='o', markersize=6)
    
    if avg_nonvul:
        nonvul_layers = sorted(avg_nonvul.keys())
        nonvul_vals = [avg_nonvul[l] for l in nonvul_layers]
        plt.plot(nonvul_layers, nonvul_vals, color='blue', linewidth=3, 
                label=f'NONVUL Average (n={len(nonvul_results)})', marker='s', markersize=6)
    
    plt.xlabel("Layer")
    plt.ylabel("L0 Norm")
    plt.title("All Sample L0 Norm Trends per Layer", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[SAVED] All samples L0 trends plot saved: {out_path}")

# === Run analysis ===
print("[START] Begin analysis...")
vul_prompts = load_prompts(VUL_PATH)
nonvul_prompts = load_prompts(NONVUL_PATH)

print(f"[INFO] Processing {len(vul_prompts)} VUL samples (optimized GPU utilization)")
vul_results, vul_predictions = process_samples_with_attr_pool(vul_prompts, tag="VUL")

print(f"[INFO] Processing {len(nonvul_prompts)} NONVUL samples (optimized GPU utilization)")
nonvul_results, nonvul_predictions = process_samples_with_attr_pool(nonvul_prompts, tag="NONVUL")

# === Compute averages ===
avg_vul = average_l0_dicts([x["l0_per_layer"] for x in vul_results])
avg_nonvul = average_l0_dicts([x["l0_per_layer"] for x in nonvul_results])

# === Calculate classification metrics ===
print("[INFO] Calculating classification metrics...")
# Use ALL predictions (including those without attribution) for metrics
metrics = calculate_classification_metrics(vul_predictions, nonvul_predictions)

print(f"\n=== Classification Summary ===")
print(f"Total samples: {metrics['total_samples']}")
print(f"Classified (non-unknown): {metrics['total_classified']}")
print(f"Unknown predictions: {metrics['total_unknown']} ({metrics['total_unknown']/metrics['total_samples']*100:.1f}%)")
print(f"  - Unknown among VUL: {metrics['unknown_vul']}")
print(f"  - Unknown among NONVUL: {metrics['unknown_nonvul']}")
print(f"\nConfusion Matrix:")
print(f"True Positive (TP): {metrics['tp']}")
print(f"False Positive (FP): {metrics['fp']}")
print(f"True Negative (TN): {metrics['tn']}")
print(f"False Negative (FN): {metrics['fn']}")
print(f"\nMetrics (on non-unknown):")
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1 score: {metrics['f1_score']:.4f}")
print(f"Overall accuracy (incl. unknown): {metrics['overall_accuracy']:.4f}")

# === Save plots ===
plot_path = os.path.join(PLOT_DIR, "l0_comparison.png")
plot_comparison(avg_vul, avg_nonvul, plot_path)

all_trends_path = os.path.join(PLOT_DIR, "l0_all_samples_trends.png")
plot_all_samples_l0_trends(vul_results, nonvul_results, all_trends_path)

# === Save JSON ===
out_json = {
    "avg_vul": avg_vul,
    "avg_nonvul": avg_nonvul,
    # Attribution counts (with L0)
    "num_vul": len(vul_results),
    "num_nonvul": len(nonvul_results),
    "num_vul_attr": len(vul_results),
    "num_nonvul_attr": len(nonvul_results),
    # Prediction counts (all samples)
    "num_vul_predictions": len(vul_predictions),
    "num_nonvul_predictions": len(nonvul_predictions),
    "classification_metrics": metrics,
    "vul_samples": vul_results,
    "nonvul_samples": nonvul_results,
    # Include all predictions explicitly
    "vul_predictions_all": vul_predictions,
    "nonvul_predictions_all": nonvul_predictions
}
json_path = os.path.join(LOG_DIR, "out.json")
with open(json_path, "w") as f:
    json.dump(out_json, f, indent=2)

print(f"\n[DONE] Results saved: {json_path}")

# === Save all model predictions ===
print("[INFO] Saving all model prediction outputs...")
all_predictions = []

# Add VUL samples predictions (ALL)
for result in vul_predictions:
    all_predictions.append({
        "idx": result.get("idx"),
        "true_label": result.get("true_label"),
        "pred_label": result.get("pred_label"),
        "model_output": result.get("model_output"),
        "sample_type": "vul"
    })

# Add NONVUL samples predictions (ALL)
for result in nonvul_predictions:
    all_predictions.append({
        "idx": result.get("idx"),
        "true_label": result.get("true_label"),
        "pred_label": result.get("pred_label"),
        "model_output": result.get("model_output"),
        "sample_type": "nonvul"
    })

# Sort by index
all_predictions.sort(key=lambda x: (x.get("sample_type", ""), x.get("idx", -1)))

# Save to JSON
with open(predictions_json_path, "w", encoding="utf-8") as f:
    json.dump(all_predictions, f, indent=2, ensure_ascii=False)

print(f"[SAVED] All predictions saved to: {predictions_json_path}")

# Also save a human-readable text format
predictions_txt_path = os.path.join(LOG_DIR, "all_predictions.txt")
with open(predictions_txt_path, "w", encoding="utf-8") as f:
    f.write("=" * 80 + "\n")
    f.write(f"Model Prediction Outputs Collection - {ts}\n")
    f.write("=" * 80 + "\n\n")
    
    f.write(f"Total Samples: {len(all_predictions)}\n")
    f.write(f"VUL Samples: {len(vul_predictions)}\n")
    f.write(f"NONVUL Samples: {len(nonvul_predictions)}\n\n")
    
    f.write("=" * 80 + "\n")
    f.write("VUL SAMPLES PREDICTIONS\n")
    f.write("=" * 80 + "\n\n")
    
    for i, pred in enumerate([p for p in all_predictions if p["sample_type"] == "vul"], 1):
        f.write(f"--- Sample {i} (idx={pred['idx']}) ---\n")
        f.write(f"True Label: {pred['true_label']}\n")
        f.write(f"Predicted Label: {pred['pred_label']}\n")
        f.write(f"Model Output:\n{pred['model_output']}\n")
        f.write("\n" + "-" * 80 + "\n\n")
    
    f.write("=" * 80 + "\n")
    f.write("NONVUL SAMPLES PREDICTIONS\n")
    f.write("=" * 80 + "\n\n")
    
    for i, pred in enumerate([p for p in all_predictions if p["sample_type"] == "nonvul"], 1):
        f.write(f"--- Sample {i} (idx={pred['idx']}) ---\n")
        f.write(f"True Label: {pred['true_label']}\n")
        f.write(f"Predicted Label: {pred['pred_label']}\n")
        f.write(f"Model Output:\n{pred['model_output']}\n")
        f.write("\n" + "-" * 80 + "\n\n")

print(f"[SAVED] Human-readable predictions saved to: {predictions_txt_path}")
print(f"\n[COMPLETE] Generated {len(all_predictions)} prediction outputs")