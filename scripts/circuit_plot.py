#!/usr/bin/env python3
# Circuit visualization & attribution plotting for specific samples
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
import sys
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR   = os.path.dirname(_SCRIPT_DIR)
_CT_PATH    = os.path.join(_ROOT_DIR, "circuit-tracer", "circuit-tracer")
if _CT_PATH not in sys.path:
    sys.path.insert(0, _CT_PATH)

from circuit_tracer.graph import prune_graph


def visualize_circuit_simple(graph, save_path,
                             node_threshold=0.85,
                             edge_threshold=0.98,
                             max_nodes_per_layer=10,
                             show_top_k_edges=50,
                             tokenizer=None):


    print(f"[VIS] start to visualize circuit...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    graph.to(device)

    print(f"[VIS] graph (node_threshold={node_threshold}, edge_threshold={edge_threshold})...")
    node_mask, edge_mask, cumulative_scores = prune_graph(graph, node_threshold, edge_threshold)

    node_mask = node_mask.cpu()
    edge_mask = edge_mask.cpu()
    cumulative_scores = cumulative_scores.cpu()
    adj = graph.adjacency_matrix.cpu().numpy()

    n_selected_features = len(graph.selected_features) if hasattr(graph, 'selected_features') else len(graph.active_features)
    n_layers = graph.cfg.n_layers
    n_pos = graph.n_pos
    n_error = n_layers * n_pos
    n_embed = n_pos
    n_logit = len(graph.logit_tokens)

    range_feat  = (0, n_selected_features)
    range_error = (n_selected_features, n_selected_features + n_error)
    range_embed = (n_selected_features + n_error, n_selected_features + n_error + n_embed)
    range_logit = (n_selected_features + n_error + n_embed, n_selected_features + n_error + n_embed + n_logit)

    print(f"[VIS] select important feature nodes...")
    important_feature_nodes = []
    for idx in range(range_feat[0], range_feat[1]):
        if node_mask[idx]:
            selected_idx = idx - range_feat[0]
            if hasattr(graph, 'selected_features'):
                actual_idx = graph.selected_features[selected_idx]
                layer, pos, feat = graph.active_features[actual_idx].tolist()
            else:
                layer, pos, feat = graph.active_features[selected_idx].tolist()
            score = cumulative_scores[idx].item()
            important_feature_nodes.append((idx, layer, pos, feat, score))

# select top nodes per layer to avoid overcrowding
    nodes_by_layer = defaultdict(list)
    for node_info in important_feature_nodes:
        idx, layer, pos, feat, score = node_info
        nodes_by_layer[layer].append(node_info)

    selected_nodes = []
    for layer in sorted(nodes_by_layer.keys()):
        layer_nodes = sorted(nodes_by_layer[layer], key=lambda x: x[4], reverse=True)
        selected_nodes.extend(layer_nodes[:max_nodes_per_layer])

    embedding_candidates = []
    for idx in range(range_embed[0], range_embed[1]):
        if node_mask[idx]:
            pos = idx - range_embed[0]
            if hasattr(graph, 'input_tokens'):
                token_id = graph.input_tokens[pos]
                if hasattr(token_id, 'item'):
                    token_id = token_id.item()
            else:
                token_id = pos

            token_text = "UNK"
            if tokenizer is not None and hasattr(graph, 'input_tokens'):
                try:
                    token_text = tokenizer.decode([token_id])
                    token_text = token_text.strip().replace('\n', '\\n')
                    if not token_text:
                        token_text = f"<{token_id}>"
                except Exception:
                    token_text = f"T{pos}"
            else:
                token_text = f"Token[{pos}]"

            embedding_candidates.append((idx, -1, pos, token_text, cumulative_scores[idx].item()))

    for idx in range(range_logit[0], range_logit[1]):
        if node_mask[idx]:
            l_idx = idx - range_logit[0]
            selected_nodes.append((idx, 999, l_idx, 0, cumulative_scores[idx].item()))
    print(f"[VIS] selected {len(selected_nodes)} important feature/logit nodes and {len(embedding_candidates)} candidate embedding nodes before filtering...")

    selected_indices = set([n[0] for n in selected_nodes])
    candidate_embedding_indices = set([n[0] for n in embedding_candidates])
    all_candidate_indices = selected_indices | candidate_embedding_indices

    important_edges = []
    for i in all_candidate_indices:
        for j in all_candidate_indices:
            if edge_mask[j, i]:  # j means "from", i means "to"
                weight = adj[j, i]
                if abs(weight) > 1e-6:
                    important_edges.append((i, j, weight))

    important_edges.sort(key=lambda x: abs(x[2]), reverse=True)
    important_edges = important_edges[:show_top_k_edges]

    edges_from_nodes = set()
    edges_to_nodes   = set()
    for src, tgt, weight in important_edges:
        edges_from_nodes.add(src)
        edges_to_nodes.add(tgt)

    filtered_embedding_nodes = []
    for node_info in embedding_candidates:
        idx = node_info[0]
        if idx in edges_from_nodes or idx in edges_to_nodes:
            filtered_embedding_nodes.append(node_info)
    print (f"[VIS] {len(filtered_embedding_nodes)} out of {len(embedding_candidates)} embedding nodes have connections to important edges and are retained for visualization, remaining {len(embedding_candidates) - len(filtered_embedding_nodes)} are filtered out as isolated nodes.")

    selected_nodes.extend(filtered_embedding_nodes)
    print (f"[VIS] total selected nodes: {len(selected_nodes)},total selected edges: {len(important_edges)}")

    G = nx.DiGraph()

    node_info_dict = {}
    for idx, layer, pos, feat_or_text, score in selected_nodes:
        if layer == -1:  
            label = str(feat_or_text)
            if len(label) > 10:
                label = label[:8] + ".."
            node_type = 'input'
        elif layer == 999:  
            label = f"Logit[{pos}]"
            node_type = 'output'
        else:  
            label = f"L{layer}\nF{feat_or_text}"
            node_type = 'feature'

        G.add_node(idx, label=label, layer=layer, node_type=node_type, score=score)
        node_info_dict[idx] = (label, layer, node_type)

    for src, tgt, weight in important_edges:
        if src in G and tgt in G:
            G.add_edge(src, tgt, weight=weight)
    print(f"[VIS] graph construction with pruning completed. Total nodes: {len(G.nodes())}, Total edges: {len(G.edges())}")

    fig, ax = plt.subplots(figsize=(20, 14))

    pos = {}
    layer_groups = defaultdict(list)
    for node in G.nodes():
        layer = G.nodes[node]['layer']
        layer_groups[layer].append(node)

    sorted_layers   = sorted(layer_groups.keys())
    layer_y_pos     = {}
    vertical_spacing = 2.5

    for i, layer in enumerate(sorted_layers):
        if layer == -1:
            layer_y_pos[layer] = 0
        elif layer == 999:
            layer_y_pos[layer] = (len(sorted_layers) - 1) * vertical_spacing
        else:
            normalized_layer = (layer + 1) / (n_layers + 1)
            layer_y_pos[layer] = normalized_layer * (len(sorted_layers) - 2) * vertical_spacing

    for layer, nodes in layer_groups.items():
        y = layer_y_pos[layer]
        num_nodes = len(nodes)
        horizontal_spacing = 2.5
        for i, node in enumerate(nodes):
            x = (i - num_nodes / 2) * horizontal_spacing
            pos[node] = (x, y)

    edges   = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    max_weight = max([abs(w) for w in weights]) if weights else 1.0

    for u, v, data in G.edges(data=True):
        weight = data['weight']
        width  = 0.3 + 3.0 * abs(weight) / max_weight
        color  = 'blue' if weight > 0 else 'red'
        alpha  = min(0.7, 0.3 + 0.6 * abs(weight) / max_weight)
        ax.annotate('', xy=pos[v], xytext=pos[u],
                    arrowprops=dict(arrowstyle='->',
                                    lw=width, color=color, alpha=alpha,
                                    connectionstyle="arc3,rad=0.1"))

    for node in G.nodes():
        x, y      = pos[node]
        node_type = G.nodes[node]['node_type']
        label     = G.nodes[node]['label']

        if node_type == 'input':
            color, shape, size = 'lightgreen', 'o', 1500
        elif node_type == 'output':
            color, shape, size = 'salmon', 's', 1500
        else:
            color, shape, size = 'lightblue', 'o', 1200

        ax.scatter(x, y, s=size, c=color, marker=shape,
                   edgecolors='black', linewidths=2, zorder=3)

        fontsize   = 9 if node_type == 'input' else 7
        fontweight = 'bold' if node_type in ['input', 'output'] else 'normal'
        ax.text(x, y, label, ha='center', va='center',
                fontsize=fontsize, fontweight=fontweight, zorder=4)

    ax.set_title(f'Circuit Visualization (Optimized Layout)\n'
                 f'{len(G.nodes())} nodes ({len(filtered_embedding_nodes)} input tokens with connections), '
                 f'{len(G.edges())} edges',
                 fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')

    all_x = [pos[n][0] for n in G.nodes()]
    all_y = [pos[n][1] for n in G.nodes()]
    if all_x and all_y:
        x_margin = (max(all_x) - min(all_x)) * 0.15 if max(all_x) != min(all_x) else 1.0
        y_margin = (max(all_y) - min(all_y)) * 0.15 if max(all_y) != min(all_y) else 1.0
        ax.set_xlim(min(all_x) - x_margin, max(all_x) + x_margin)
        ax.set_ylim(min(all_y) - y_margin, max(all_y) + y_margin)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightgreen', edgecolor='black', label='Input Tokens (with connections)'),
        Patch(facecolor='lightblue',  edgecolor='black', label='Feature Nodes'),
        Patch(facecolor='salmon',     edgecolor='black', label='Output Logits'),
        Patch(facecolor='blue',  alpha=0.6, label='Positive Weight'),
        Patch(facecolor='red',   alpha=0.6, label='Negative Weight'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.9)

    plt.tight_layout()

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[VIS] Circuit visualization saved to: {save_path}")

    png_path = save_path.replace('.pdf', '.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"[VIS] PNG version saved to: {png_path}")

    plt.close()

    stats = {
        'total_nodes':              len(selected_nodes),
        'feature_nodes':            len([n for n in selected_nodes if -1 < n[1] < 999]),
        'input_nodes':              len(filtered_embedding_nodes),
        'input_nodes_filtered_out': len(embedding_candidates) - len(filtered_embedding_nodes),
        'output_nodes':             len([n for n in selected_nodes if n[1] == 999]),
        'edges':                    len(important_edges),
        'layers_represented':       len(set([n[1] for n in selected_nodes if -1 < n[1] < 999])),
    }

    return stats


if __name__ == '__main__':
    import json
    import gc
    from collections import defaultdict
    from datetime import datetime
    from transformers import AutoTokenizer
    from circuit_tracer.replacement_model import ReplacementModel
    from circuit_tracer.attribution.attribute import attribute
    from circuit_tracer.utils.create_graph_files import create_graph_files

    OUTPUT_BASE = os.environ.get("LLMVUL_OUTPUT_DIR", os.path.join(_ROOT_DIR, "out"))

    TARGET_CWES = ["CWE-787", "CWE-476", "CWE-125", "CWE-416", "CWE-119", "CWE-190"]
    CWE_DESCRIPTIONS = {
        "CWE-787": "Buffer Overflow (Out-of-bounds Write)",
        "CWE-476": "NULL Pointer Dereference",
        "CWE-125": "Out-of-bounds Read",
        "CWE-416": "Use After Free",
        "CWE-119": "Memory Buffer Operations",
        "CWE-190": "Integer Overflow",
    }

    VUL_PATH   = os.path.join(_ROOT_DIR, "data", "primevul236.jsonl")
    MODEL_PATH = "Chun9622/llmvul-finetuned-gemma"
    DEVICE     = "cuda:0" if torch.cuda.is_available() else "cpu"

    MAX_FEATURE_NODES   = 5000
    NODE_THRESHOLD      = 0.70
    EDGE_THRESHOLD      = 0.85
    SHOW_TOP_K_EDGES    = 150
    MAX_NODES_PER_LAYER = 20

    ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_DIR     = os.path.join(OUTPUT_BASE, "log",   f"cwe_circuits_{ts}")
    CIRCUIT_DIR = os.path.join(OUTPUT_BASE, "plots", f"cwe_circuits_{ts}", "circuits")
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CIRCUIT_DIR, exist_ok=True)

    _log_file  = open(os.path.join(LOG_DIR, f"cwe_analysis_{ts}.txt"), "w")
    sys.stdout = _log_file
    sys.stderr = _log_file

    print("=" * 80)
    print("CWE-Based Circuit Visualization Analysis")
    print("=" * 80)
    print(f"Target CWEs : {TARGET_CWES}")
    print(f"Output Dir  : {CIRCUIT_DIR}")
    print("=" * 80)

    def _patch_model_loading():
        import transformer_lens.loading_from_pretrained as _loading
        _orig = _loading.get_official_model_name
        _loading.get_official_model_name = (
            lambda m: "google/gemma-2-2b" if m == MODEL_PATH else _orig(m)
        )

    def _patch_model_config_loading():
        import transformer_lens.loading_from_pretrained as _loading
        _orig = _loading.get_pretrained_model_config
        def _patched(m, **kw):
            if m == MODEL_PATH:
                from transformers import AutoConfig
                return AutoConfig.from_pretrained(m)
            return _orig(m, **kw)
        _loading.get_pretrained_model_config = _patched

    _patch_model_loading()
    _patch_model_config_loading()

    print("\n[INFO] Loading Model & Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rm = ReplacementModel.from_pretrained(
        MODEL_PATH, transcoder_set="gemma", device=DEVICE, dtype=torch.float16,
    )
    rm.eval()
    print("[INFO] Model Loaded Successfully.")

    def _clear_gpu():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def _extract_label(text):
        t = text.lower()
        if 'vulnerable' in t and 'not vulnerable' not in t: return "vul"
        if 'safe' in t and 'not safe' not in t: return "nonvul"
        return "unknown"

    def _load_samples_by_cwe(jsonl_path):
        cwe_samples = defaultdict(list)
        with open(jsonl_path) as f:
            for line in f:
                try:
                    obj  = json.loads(line)
                    cwes = obj.get("cwe", [])
                    code = obj.get("func", "").strip()
                    if not code or not cwes:
                        continue
                    if isinstance(cwes, str):
                        cwes = [cwes]
                    for cwe in cwes:
                        if cwe in TARGET_CWES:
                            if len(code) > 1000:
                                code = code[:1000] + "\n// ... (truncated)"
                            prompt = (f"Code: {code}\n\n"
                                      f"Question: Is this code safe or vulnerable?\nAnswer:")
                            cwe_samples[cwe].append({
                                "idx": obj.get("idx", -1), "cwe": cwe,
                                "prompt": prompt, "code_length": len(code),
                            })
                except Exception:
                    pass
        return cwe_samples

    def _select_representative(samples):
        if not samples: return None
        ss = sorted(samples, key=lambda x: abs(x['code_length'] - 400))
        for s in ss:
            if 200 <= s['code_length'] <= 600:
                return s
        return ss[0]

    def _generate_circuit(cwe, sample):
        idx, prompt = sample['idx'], sample['prompt']
        print(f"\n{'='*80}\nProcessing CWE : {cwe}")
        print(f"Description    : {CWE_DESCRIPTIONS.get(cwe, 'Unknown')}")
        print(f"Sample ID      : {idx}  |  Code length: {sample['code_length']} chars\n{'='*80}")

        print("Running model prediction...")
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            out  = rm.generate(inputs.input_ids, max_new_tokens=50, do_sample=False)
            pred = _extract_label(tokenizer.decode(out[0], skip_special_tokens=True))
        print(f"  Prediction: {pred}")

        # Step 2: attribute
        print("Running attribution analysis...")
        _clear_gpu()
        try:
            with torch.enable_grad():
                g = attribute(prompt=prompt, model=rm, max_n_logits=3,
                              batch_size=1, max_feature_nodes=MAX_FEATURE_NODES, verbose=False)
            print(f"  Active features found: {len(g.active_features) if hasattr(g,'active_features') else 0}")

            # Step 3: visualize
            print("Generating circuit visualisation...")
            cwe_clean  = cwe.replace("CWE-", "")
            desc_clean = (CWE_DESCRIPTIONS.get(cwe, "Unknown")
                          .replace(" ", "_").replace("/", "_")
                          .replace("(", "").replace(")", ""))
            fname     = f"circuit_{cwe_clean}_{desc_clean}_sample{idx}"
            save_path = os.path.join(CIRCUIT_DIR, fname + ".pdf")

            stats = visualize_circuit_simple(
                g, save_path,
                node_threshold=NODE_THRESHOLD,
                edge_threshold=EDGE_THRESHOLD,
                max_nodes_per_layer=MAX_NODES_PER_LAYER,
                show_top_k_edges=SHOW_TOP_K_EDGES,
                tokenizer=tokenizer,
            )

            try:
                g.to("cpu")
                create_graph_files(
                    graph_or_path=g, slug=fname, scan=getattr(g, "scan", "gemma"),
                    output_path=CIRCUIT_DIR,
                    node_threshold=NODE_THRESHOLD, edge_threshold=EDGE_THRESHOLD,
                )
                print(f" JSON saved: {fname}.json")
            except Exception as _je:
                print(f"  [WARN] JSON save failed: {_je}")

            print(f"  Circuit saved: {fname}.pdf/.png/.txt")
            return {"cwe": cwe, "idx": idx, "success": True,
                    "files": {"pdf": save_path,
                              "txt": save_path.replace(".pdf", ".txt"),
                              "json": os.path.join(CIRCUIT_DIR, fname + ".json")},
                    "stats": stats}

        except Exception as e:
            import traceback
            print(f"  ✗ Error: {e}")
            traceback.print_exc()
            return {"cwe": cwe, "idx": idx, "success": False, "error": str(e)}

    print("\nLoading and organising samples by CWE...")
    cwe_samples = _load_samples_by_cwe(VUL_PATH)
    for cwe in TARGET_CWES:
        count = len(cwe_samples.get(cwe, []))
        print(f"  - {cwe} ({CWE_DESCRIPTIONS.get(cwe, 'Unknown')}): {count} samples")

    print("\nSelecting representative samples...")
    selected_samples = {}
    for cwe in TARGET_CWES:
        s = _select_representative(cwe_samples.get(cwe, []))
        if s:
            selected_samples[cwe] = s
            print(f"  ✓ {cwe}: Sample {s['idx']} (length={s['code_length']})")
        else:
            print(f"  ✗ {cwe}: No samples found")
    print(f"\n[INFO] Selected {len(selected_samples)} samples for visualisation")

    print("\nGenerating circuit visualisations...")
    results = []
    for i, (cwe, sample) in enumerate(selected_samples.items(), 1):
        print(f"\n[{i}/{len(selected_samples)}] Processing {cwe}...")
        results.append(_generate_circuit(cwe, sample))
        _clear_gpu()

    successful = [r for r in results if r.get("success")]
    failed     = [r for r in results if not r.get("success")]
    print(f"\n{'='*80}\nANALYSIS COMPLETE\n{'='*80}")
    print(f"\nSuccessfully generated: {len(successful)}/{len(results)} circuits")
    for r in successful:
        print(f"  - {r['cwe']} ({CWE_DESCRIPTIONS.get(r['cwe'], '')}), Sample {r['idx']}")
    for r in failed:
        print(f"  ✗ {r['cwe']}: {r.get('error', 'unknown error')}")

    summary_path = os.path.join(LOG_DIR, "CWE_CIRCUITS_SUMMARY.md")
    with open(summary_path, 'w') as f:
        f.write("# CWE-Based Circuit Visualisation Summary\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Output Directory:** `{CIRCUIT_DIR}`\n\n")
        f.write(f"## Generated Circuits\n\nSuccessfully generated {len(successful)} / {len(results)}.\n\n")
        for r in successful:
            cwe  = r['cwe']
            desc = CWE_DESCRIPTIONS.get(cwe, '')
            f.write(f"### {cwe}: {desc}\n\n")
            f.write(f"- **Sample ID:** {r['idx']}\n")
            f.write(f"- **PDF:** `{os.path.basename(r['files']['pdf'])}`\n")
            f.write(f"- **Statistics:** `{os.path.basename(r['files']['txt'])}`\n")
            if r.get('stats'):
                st = r['stats']
                f.write(f"- **Nodes:** {st.get('total_nodes','?')}  "
                        f"**Edges:** {st.get('edges','?')}\n")
            f.write("\n")

    print(f"\n[INFO] Summary: {summary_path}")
    print("[DONE] All processing complete!")
    _log_file.close()
