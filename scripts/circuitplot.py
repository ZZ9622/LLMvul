#!/usr/bin/env python3
"""Circuit plot: attribution and graph visualization. Includes visualize_circuit_simple for diagram generation."""
import os
import sys
import json
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
from datetime import datetime
from transformers import AutoTokenizer
import warnings

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import config
config.setup_circuit_tracer()
from circuit_tracer.replacement_model import ReplacementModel
from circuit_tracer.attribution.attribute import attribute
from circuit_tracer.graph import prune_graph


def visualize_circuit_simple(graph, save_path,
                             node_threshold=0.85,
                             edge_threshold=0.98,
                             max_nodes_per_layer=10,
                             show_top_k_edges=50,
                             tokenizer=None):
    """Circuit diagram: token labels, input nodes with edges only."""
    print("[VIS] Generating circuit diagram...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    graph.to(device)
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

    range_feat = (0, n_selected_features)
    range_error = (n_selected_features, n_selected_features + n_error)
    range_embed = (n_selected_features + n_error, n_selected_features + n_error + n_embed)
    range_logit = (n_selected_features + n_error + n_embed, n_selected_features + n_error + n_embed + n_logit)

    important_feature_nodes = []
    for idx in range(range_feat[0], range_feat[1]):
        if node_mask[idx]:
            selected_idx = idx - range_feat[0]
            if hasattr(graph, 'selected_features'):
                layer, pos, feat = graph.active_features[graph.selected_features[selected_idx]].tolist()
            else:
                layer, pos, feat = graph.active_features[selected_idx].tolist()
            score = cumulative_scores[idx].item()
            important_feature_nodes.append((idx, layer, pos, feat, score))

    nodes_by_layer = defaultdict(list)
    for node_info in important_feature_nodes:
        nodes_by_layer[node_info[1]].append(node_info)
    selected_nodes = []
    for layer in sorted(nodes_by_layer.keys()):
        layer_nodes = sorted(nodes_by_layer[layer], key=lambda x: x[4], reverse=True)
        selected_nodes.extend(layer_nodes[:max_nodes_per_layer])

    embedding_candidates = []
    for idx in range(range_embed[0], range_embed[1]):
        if node_mask[idx]:
            pos = idx - range_embed[0]
            token_id = graph.input_tokens[pos].item() if hasattr(graph, 'input_tokens') and hasattr(graph.input_tokens[pos], 'item') else pos
            token_text = "UNK"
            if tokenizer is not None and hasattr(graph, 'input_tokens'):
                try:
                    token_text = tokenizer.decode([token_id]).strip().replace('\n', '\\n') or f"<{token_id}>"
                except Exception:
                    token_text = f"T{pos}"
            else:
                token_text = f"Token[{pos}]"
            embedding_candidates.append((idx, -1, pos, token_text, cumulative_scores[idx].item()))

    for idx in range(range_logit[0], range_logit[1]):
        if node_mask[idx]:
            l_idx = idx - range_logit[0]
            selected_nodes.append((idx, 999, l_idx, 0, cumulative_scores[idx].item()))

    selected_indices = set(n[0] for n in selected_nodes)
    candidate_embedding_indices = set(n[0] for n in embedding_candidates)
    all_candidate_indices = selected_indices | candidate_embedding_indices

    important_edges = []
    for i in all_candidate_indices:
        for j in all_candidate_indices:
            if edge_mask[j, i]:
                weight = adj[j, i]
                if abs(weight) > 1e-6:
                    important_edges.append((i, j, weight))
    important_edges.sort(key=lambda x: abs(x[2]), reverse=True)
    important_edges = important_edges[:show_top_k_edges]

    edges_from_nodes = set(e[0] for e in important_edges)
    edges_to_nodes = set(e[1] for e in important_edges)
    filtered_embedding_nodes = [n for n in embedding_candidates if n[0] in edges_from_nodes or n[0] in edges_to_nodes]
    selected_nodes.extend(filtered_embedding_nodes)

    G = nx.DiGraph()
    node_info_dict = {}
    for idx, layer, pos, feat_or_text, score in selected_nodes:
        if layer == -1:
            label = str(feat_or_text)[:10] + (".." if len(str(feat_or_text)) > 10 else "")
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

    fig, ax = plt.subplots(figsize=(20, 14))
    layer_groups = defaultdict(list)
    for node in G.nodes():
        layer_groups[G.nodes[node]['layer']].append(node)
    sorted_layers = sorted(layer_groups.keys())
    vertical_spacing = 2.5
    layer_y_pos = {}
    for i, layer in enumerate(sorted_layers):
        if layer == -1:
            layer_y_pos[layer] = 0
        elif layer == 999:
            layer_y_pos[layer] = (len(sorted_layers) - 1) * vertical_spacing
        else:
            normalized_layer = (layer + 1) / (n_layers + 1)
            layer_y_pos[layer] = normalized_layer * (len(sorted_layers) - 2) * vertical_spacing

    pos = {}
    horizontal_spacing = 2.5
    for layer, nodes in layer_groups.items():
        y = layer_y_pos[layer]
        num_nodes = len(nodes)
        for i, node in enumerate(nodes):
            x = (i - num_nodes / 2) * horizontal_spacing
            pos[node] = (x, y)

    weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max((abs(w) for w in weights), default=1.0)
    for u, v, data in G.edges(data=True):
        weight = data['weight']
        width = 0.3 + 3.0 * abs(weight) / max_weight
        color = 'blue' if weight > 0 else 'red'
        alpha = min(0.7, 0.3 + 0.6 * abs(weight) / max_weight)
        ax.annotate('', xy=pos[v], xytext=pos[u],
                    arrowprops=dict(arrowstyle='->', lw=width, color=color, alpha=alpha, connectionstyle="arc3,rad=0.1"))

    for node in G.nodes():
        x, y = pos[node]
        node_type = G.nodes[node]['node_type']
        label = G.nodes[node]['label']
        if node_type == 'input':
            color, shape, size = 'lightgreen', 'o', 1500
        elif node_type == 'output':
            color, shape, size = 'salmon', 's', 1500
        else:
            color, shape, size = 'lightblue', 'o', 1200
        ax.scatter(x, y, s=size, c=color, marker=shape, edgecolors='black', linewidths=2, zorder=3)
        fontsize = 9 if node_type == 'input' else 7
        fontweight = 'bold' if node_type in ['input', 'output'] else 'normal'
        ax.text(x, y, label, ha='center', va='center', fontsize=fontsize, fontweight=fontweight, zorder=4)

    ax.set_title(f'Circuit Visualization\n{len(G.nodes())} nodes, {len(G.edges())} edges', fontsize=16, fontweight='bold', pad=20)
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
        Patch(facecolor='lightgreen', edgecolor='black', label='Input Tokens'),
        Patch(facecolor='lightblue', edgecolor='black', label='Feature Nodes'),
        Patch(facecolor='salmon', edgecolor='black', label='Output Logits'),
        Patch(facecolor='blue', alpha=0.6, label='Positive Weight'),
        Patch(facecolor='red', alpha=0.6, label='Negative Weight')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[VIS] Saved: {save_path}")
    png_path = save_path.replace('.pdf', '.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"[VIS] Saved: {png_path}")
    plt.close()
    return {
        'total_nodes': len(selected_nodes),
        'feature_nodes': len([n for n in selected_nodes if -1 < n[1] < 999]),
        'input_nodes': len(filtered_embedding_nodes),
        'output_nodes': len([n for n in selected_nodes if n[1] == 999]),
        'edges': len(important_edges),
    }


def main():
    """Run attribution + visualization for target samples."""
    global visualize_graph
    try:
        if _SCRIPT_DIR not in sys.path:
            sys.path.insert(0, _SCRIPT_DIR)
        from visualize_custom import visualize_graph
    except ImportError:
        visualize_graph = None

    TARGET_IDS = [196316, 90797, 205736, 220195]
    VUL_PATH = config.VUL_JSONL
    NONVUL_PATH = config.NONVUL_JSONL
    MODEL_NAME = config.MODEL_NAME
    DEVICE = config.DEVICE

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_DIR = os.path.join(config.LOG_DIR, ts) if config.LOG_DIR else ""
    PLOT_DIR = os.path.join(config.PLOT_DIR, ts) if config.PLOT_DIR else ""
    CIRCUIT_DIR = os.path.join(PLOT_DIR, "circuits") if PLOT_DIR else ""
    if LOG_DIR:
        os.makedirs(LOG_DIR, exist_ok=True)
    if CIRCUIT_DIR:
        os.makedirs(CIRCUIT_DIR, exist_ok=True)

    MAX_FEATURE_NODES = 5000
    EDGE_THRESHOLD = 0.01
    NODE_THRESHOLD = 0.2
    SHOW_TOP_K_EDGES = 300

    if LOG_DIR:
        sys.stdout = open(os.path.join(LOG_DIR, f"log_{ts}.txt"), "w")
        sys.stderr = sys.stdout

    print("[INFO] Target IDs:", TARGET_IDS)
    print("[INFO] Loading Model & Tokenizer...")

    def patch_model_loading():
        import transformer_lens.loading_from_pretrained as loading
        original = loading.get_official_model_name
        loading.get_official_model_name = (
            lambda model_name: "google/gemma-2-2b" if model_name == MODEL_NAME else original(model_name)
        )

    def patch_model_config_loading():
        import transformer_lens.loading_from_pretrained as loading
        original = loading.get_pretrained_model_config
        def patched(model_name, **kwargs):
            if model_name == MODEL_NAME:
                from transformers import AutoConfig
                return AutoConfig.from_pretrained(model_name)
            return original(model_name, **kwargs)
        loading.get_pretrained_model_config = patched

    patch_model_loading()
    patch_model_config_loading()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    rm = ReplacementModel.from_pretrained(
        MODEL_NAME,
        transcoder_set="gemma",
        device=DEVICE,
        torch_dtype=torch.float16
    )
    rm.eval()
    print("[INFO] Model Loaded.")

    def clear_gpu_cache():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def extract_label(text):
        t = text.lower()
        if 'vulnerable' in t and 'not vulnerable' not in t:
            return "vul"
        if 'safe' in t and 'not safe' not in t:
            return "nonvul"
        return "unknown"

    def load_and_filter_prompts(jsonl_path, target_ids):
        samples = []
        with open(jsonl_path, "r") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    idx = obj.get("idx", -1)
                    if idx in target_ids:
                        code = obj.get("func", "").strip()
                        prompt = f"Code: {code}\n\nQuestion: Is this code safe or vulnerable?\nAnswer:"
                        samples.append({
                            "idx": idx,
                            "true_label": "vul" if obj.get("target") == 1 else "nonvul",
                            "prompt": prompt
                        })
                except Exception:
                    pass
        return samples

    print("[INFO] Loading and filtering data...")
    vul_samples = load_and_filter_prompts(VUL_PATH, TARGET_IDS)
    nonvul_samples = load_and_filter_prompts(NONVUL_PATH, TARGET_IDS)
    all_targets = vul_samples + nonvul_samples
    print(f"[INFO] Found {len(all_targets)} samples.")

    for i, sample in enumerate(all_targets):
        idx = sample['idx']
        tag = sample['true_label'].upper()
        print(f"\n[{i+1}/{len(all_targets)}] Sample {idx} ({tag})...")
        inputs = tokenizer(sample['prompt'], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            output = rm.generate(inputs.input_ids, max_new_tokens=50, do_sample=False)
            output_text = tokenizer.decode(output[0], skip_special_tokens=True)
            pred_label = extract_label(output_text)
        print(f"   True: {sample['true_label']} | Pred: {pred_label}")
        print("   Running attribution...")
        clear_gpu_cache()
        try:
            with torch.enable_grad():
                g = attribute(
                    prompt=sample['prompt'],
                    model=rm,
                    max_n_logits=3,
                    batch_size=1,
                    max_feature_nodes=MAX_FEATURE_NODES,
                    verbose=False
                )
            if visualize_graph and CIRCUIT_DIR:
                filename = f"circuit_{tag}_{idx}.pdf"
                save_path = os.path.join(CIRCUIT_DIR, filename)
                result = visualize_graph(
                    g, save_path,
                    node_threshold=NODE_THRESHOLD,
                    edge_threshold=EDGE_THRESHOLD,
                    max_nodes_per_layer=15,
                    show_top_k_edges=SHOW_TOP_K_EDGES,
                    save_json=True,
                    tokenizer=tokenizer
                )
                if result.get("success"):
                    print(f"   [SUCCESS] Graph saved to {save_path}")
                else:
                    print("   [FAIL] Visualization returned false.")
            else:
                print("   [WARN] Visualization skipped (no output dir or library).")
        except Exception as e:
            print(f"   [ERROR] {e}")
            import traceback
            traceback.print_exc()

    print("\n[DONE] Circuit plot completed.")
    if CIRCUIT_DIR:
        print(f"[INFO] Output: {CIRCUIT_DIR}")


if __name__ == "__main__":
    main()
