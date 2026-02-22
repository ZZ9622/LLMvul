#!/usr/bin/env python3
"""
Draw circuit diagrams with circles/nodes from circuit-tracer JSON files.
Usage: python draw_circuit_from_json.py <json_file_or_dir>
"""
import os, sys, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────
TOP_N_NODES       = 60    # keep top-N nodes by |influence|
MIN_EDGE_WEIGHT   = 0.5   # only draw edges above this absolute weight
FIGSIZE           = (22, 14)
DPI               = 150

# Node type styles
NODE_STYLES = {
    "embed":                   dict(color="#2ecc71", marker="D", size=260, zorder=4),   # green diamond
    "cross layer transcoder":  dict(color="#3498db", marker="o", size=200, zorder=3),   # blue circle
    "logit":                   dict(color="#e74c3c", marker="s", size=280, zorder=4),   # red square
    "error":                   dict(color="#f39c12", marker="^", size=160, zorder=2),   # orange triangle
    "other":                   dict(color="#95a5a6", marker="o", size=140, zorder=1),
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_json(path):
    with open(path) as f:
        return json.load(f)

def layer_order(layer_str):
    """Return numeric sort key: embed=-1, logit=999, else int."""
    try:
        return int(layer_str)
    except (ValueError, TypeError):
        s = str(layer_str).lower()
        if "embed" in s:  return -1
        if "logit" in s:  return 999
        return 500

def clerp_label(node, tokens):
    """Short readable label for a node."""
    c = node.get("clerp", "")
    if c:
        return c[:20]
    ft = node.get("feature_type", "")
    if "embed" in ft:
        idx = node.get("ctx_idx", "")
        tok = tokens[idx] if tokens and isinstance(idx, int) and idx < len(tokens) else ""
        return f"E:{tok}" if tok else f"E{idx}"
    if node.get("is_target_logit"):
        return "→logit"
    return f"L{node.get('layer','')}F{node.get('feature','')}".replace("None","?")

def draw_circuit(json_path, out_png=None, top_n=TOP_N_NODES,
                 min_edge=MIN_EDGE_WEIGHT, title=""):
    data    = load_json(json_path)
    nodes   = data["nodes"]
    links   = data["links"]
    meta    = data.get("metadata", {})
    tokens  = meta.get("prompt_tokens", [])
    prompt  = meta.get("prompt", "")

    # ── 1. Select top-N nodes by |influence| ─────────────────────────────────
    nodes_sorted = sorted(nodes, key=lambda n: abs(n.get("influence") or 0), reverse=True)

    # Always include all embed & logit nodes
    priority   = [n for n in nodes_sorted if
                  "embed" in str(n.get("feature_type","")).lower()
                  or n.get("is_target_logit")]
    feature_ns = [n for n in nodes_sorted if n not in priority]
    selected   = priority + feature_ns[:max(0, top_n - len(priority))]

    sel_ids = {n["node_id"] for n in selected}

    # ── 2. Filter edges ───────────────────────────────────────────────────────
    sel_links = [e for e in links
                 if e["source"] in sel_ids and e["target"] in sel_ids
                 and abs(e["weight"]) >= min_edge]

    # ── 3. Layout: y = layer, x = ctx_idx spread ─────────────────────────────
    layers_used = sorted({layer_order(n.get("layer")) for n in selected})
    layer_to_y  = {l: i for i, l in enumerate(layers_used)}

    # group nodes by layer
    by_layer = defaultdict(list)
    for n in selected:
        by_layer[layer_order(n.get("layer"))].append(n)

    # assign x within layer using ctx_idx (spread evenly)
    pos = {}
    for layer_k, ns in by_layer.items():
        ns_sorted = sorted(ns, key=lambda n: n.get("ctx_idx", 0) or 0)
        if len(ns_sorted) == 1:
            xs = [0.5]
        else:
            xs = np.linspace(0.05, 0.95, len(ns_sorted))
        for x, n in zip(xs, ns_sorted):
            pos[n["node_id"]] = (x, layer_to_y[layer_k])

    # ── 4. Draw ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.7, max(layer_to_y.values()) + 0.7)
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("#ffffff")
    ax.axis("off")

    # Draw layer bands
    for ly, yi in layer_to_y.items():
        if ly == -1:
            label = "Embed"
        elif ly == 999:
            label = "Logit"
        else:
            label = f"Layer {ly}"
        ax.axhspan(yi - 0.45, yi + 0.45, alpha=0.06,
                   color="#3498db" if ly not in (-1, 999) else "#2ecc71" if ly == -1 else "#e74c3c")
        ax.text(-0.04, yi, label, ha="right", va="center", fontsize=7, color="#555")

    # Draw edges
    pos_vals = pos
    for e in sel_links:
        s, t = e["source"], e["target"]
        if s not in pos_vals or t not in pos_vals:
            continue
        x0, y0 = pos_vals[s]
        x1, y1 = pos_vals[t]
        w = e["weight"]
        color  = "#2980b9" if w >= 0 else "#c0392b"
        alpha  = min(0.8, 0.15 + abs(w) / 20.0)
        lw     = min(2.5, 0.3 + abs(w) / 15.0)
        ax.annotate("",
            xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="->" ,
                color=color, alpha=alpha, lw=lw,
                connectionstyle="arc3,rad=0.05"
            ), zorder=2)

    # Draw nodes
    node_map = {n["node_id"]: n for n in selected}
    for nid, (x, y) in pos_vals.items():
        n  = node_map[nid]
        ft = str(n.get("feature_type", "other")).lower()
        if n.get("is_target_logit"):
            ft = "logit"
        elif "embed" in ft:
            ft = "embed"
        elif "error" in ft:
            ft = "error"
        elif "cross layer" in ft or "transcoder" in ft:
            ft = "cross layer transcoder"
        else:
            ft = "other"

        style = NODE_STYLES.get(ft, NODE_STYLES["other"])
        inf   = abs(n.get("influence") or 0)
        size  = style["size"] * (0.6 + 0.8 * min(inf / 20.0, 1.0))

        ax.scatter(x, y, s=size, c=style["color"], marker=style["marker"],
                   edgecolors="white", linewidths=0.8, zorder=style["zorder"])

        label = clerp_label(n, tokens)
        ax.text(x, y - 0.30, label, ha="center", va="top",
                fontsize=5.5, color="#222", zorder=5,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))

    # ── 5. Legend + title ─────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(color="#2ecc71", label="Input embedding"),
        mpatches.Patch(color="#3498db", label="Transcoder feature"),
        mpatches.Patch(color="#e74c3c", label="Output logit"),
        mpatches.Patch(color="#f39c12", label="Error node"),
        plt.Line2D([0],[0], color="#2980b9", lw=1.5, label="Positive weight"),
        plt.Line2D([0],[0], color="#c0392b", lw=1.5, label="Negative weight"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=8,
              framealpha=0.9, edgecolor="#ccc")

    disp_title = title or os.path.splitext(os.path.basename(json_path))[0]
    ax.set_title(f"Circuit Diagram: {disp_title}\n"
                 f"(Top {len(selected)} nodes · {len(sel_links)} edges shown · "
                 f"threshold |w|≥{min_edge})",
                 fontsize=10, pad=12)

    # Prompt snippet
    snippet = prompt[:120].replace("\n", " ") + ("…" if len(prompt) > 120 else "")
    fig.text(0.01, 0.01, f"Prompt: {snippet}", fontsize=7, color="#777",
             wrap=True, va="bottom")

    # ── 6. Save ───────────────────────────────────────────────────────────────
    if out_png is None:
        out_png = os.path.splitext(json_path)[0] + "_circles.png"
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {out_png}")
    return out_png

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw circuit diagrams from JSON files")
    parser.add_argument("path", help="JSON file or directory containing JSON files")
    parser.add_argument("--top_n", type=int, default=TOP_N_NODES,
                        help="Number of top nodes to display")
    parser.add_argument("--min_edge", type=float, default=MIN_EDGE_WEIGHT,
                        help="Minimum absolute edge weight to draw")
    args = parser.parse_args()

    target = args.path
    if os.path.isdir(target):
        import glob
        jsons = sorted(glob.glob(os.path.join(target, "*.json")))
        jsons = [j for j in jsons if "graph-metadata" not in j]
        print(f"[INFO] Found {len(jsons)} JSON files in {target}")
        for jf in jsons:
            try:
                draw_circuit(jf, top_n=args.top_n, min_edge=args.min_edge)
            except Exception as e:
                print(f"[ERROR] {jf}: {e}")
    elif os.path.isfile(target):
        draw_circuit(target, top_n=args.top_n, min_edge=args.min_edge)
    else:
        print(f"[ERROR] Not found: {target}")
        sys.exit(1)
