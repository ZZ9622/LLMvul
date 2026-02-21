
import torch
import numpy as np
import os
# ── Repository root & output directory (auto-detected) ───────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.dirname(_SCRIPT_DIR)           # LLMvul/
OUTPUT_BASE = os.environ.get(
    "LLMVUL_OUTPUT_DIR", os.path.join(ROOT_DIR, "out")
)
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "circuit-tracer", "circuit-tracer"))

from circuit_tracer.graph import prune_graph
from circuit_tracer.utils.create_graph_files import create_graph_files

try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from circuit_plot import visualize_circuit_simple
    HAS_CIRCUIT_PLOT = True
    print("[INFO] Successfully imported circuit_plot module")
except ImportError as e:
    print(f"[WARN] Could not import circuit_plot: {e}, visual graph will be skipped")
    HAS_CIRCUIT_PLOT = False

def visualize_graph(graph, save_path, feature_labels=None, open_in_browser=False, 
                    node_threshold=0.85, edge_threshold=0.98, save_json=True,
                    max_nodes_per_layer=10, show_top_k_edges=50, tokenizer=None):
    """
    Visualize graph using circuit-tracer and generate simplified circuit diagram.
    
    This function:
    1. Prunes the graph to keep only important nodes and edges
    2. Generates simplified circuit diagram (PDF/PNG) - shows feature nodes only
    3. Generates detailed statistics report (.txt file)
    4. Generates JSON file (for official web server)
    
    Args:
        graph: The Graph object from circuit_tracer
        save_path: Base path to save files
        node_threshold: Threshold for node pruning (0-1, higher = sparser)
        edge_threshold: Threshold for edge pruning (0-1, higher = sparser)
        max_nodes_per_layer: Max feature nodes to show per layer
        show_top_k_edges: Number of top edges to display
    """
    try:
        print(f"[VIS] Analyzing graph with {len(graph.active_features)} active features...")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        graph.to(device)
        
        print(f"[VIS] Pruning graph (node_threshold={node_threshold}, edge_threshold={edge_threshold})...")
        print(f"[VIS] Note: Error nodes represent residual stream parts not explained by transcoder")
        node_mask, edge_mask, cumulative_scores = prune_graph(graph, node_threshold, edge_threshold)
    
        node_mask_cpu = node_mask.cpu()
        edge_mask_cpu = edge_mask.cpu()
        cumulative_scores_cpu = cumulative_scores.cpu()
        
        n_features = len(graph.active_features) if hasattr(graph, 'active_features') else 0
        n_selected_features = len(graph.selected_features) if hasattr(graph, 'selected_features') else n_features
        n_layers = graph.cfg.n_layers
        n_pos = graph.n_pos
        n_error = n_layers * n_pos
        n_embed = n_pos
        n_logit = len(graph.logit_tokens)
        
        total_nodes = len(node_mask_cpu)
        active_nodes = node_mask_cpu.sum().item()
        
        range_feat = (0, n_selected_features)
        range_error = (n_selected_features, n_selected_features + n_error)
        range_embed = (n_selected_features + n_error, n_selected_features + n_error + n_embed)
        range_logit = (n_selected_features + n_error + n_embed, total_nodes)
        
        active_feature_nodes = node_mask_cpu[range_feat[0]:range_feat[1]].sum().item()
        active_error_nodes = node_mask_cpu[range_error[0]:range_error[1]].sum().item()
        active_embed_nodes = node_mask_cpu[range_embed[0]:range_embed[1]].sum().item()
        active_logit_nodes = node_mask_cpu[range_logit[0]:range_logit[1]].sum().item()
        
        total_edges = edge_mask_cpu.sum().item()
        
        layer_distribution = {}
        if active_feature_nodes > 0:
            for idx in range(range_feat[0], range_feat[1]):
                if node_mask_cpu[idx]:
                    selected_idx = idx - range_feat[0]
                    if hasattr(graph, 'active_features') and hasattr(graph, 'selected_features'):
                        layer = graph.active_features[graph.selected_features[selected_idx]][0].item()
                        layer_distribution[layer] = layer_distribution.get(layer, 0) + 1
        
        base_path = os.path.splitext(save_path)[0]
        txt_path = base_path + ".txt"
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("Circuit Analysis Report\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Input: {graph.input_string[:200]}...\n\n")
            
            f.write("Node Statistics:\n")
            f.write(f"  Total nodes: {total_nodes}\n")
            f.write(f"  Active nodes after pruning: {active_nodes} ({active_nodes/total_nodes*100:.1f}%)\n")
            f.write(f"    - Feature nodes: {active_feature_nodes}\n")
            f.write(f"    - Error nodes: {active_error_nodes}\n")
            f.write(f"      [Note: Error nodes represent unexplained residual stream]\n")
            f.write(f"    - Embedding nodes: {active_embed_nodes}\n")
            f.write(f"    - Logit nodes: {active_logit_nodes}\n\n")
            
            f.write("Edge Statistics:\n")
            f.write(f"  Active edges: {total_edges}\n\n")
            
            if layer_distribution:
                f.write("Layer Distribution of Active Features:\n")
                for layer in sorted(layer_distribution.keys()):
                    f.write(f"  Layer {layer}: {layer_distribution[layer]} nodes\n")
                f.write("\n")
            
            f.write("Top Influential Feature Nodes:\n")
            f.write("[Note: Showing feature nodes only, excluding Error nodes]\n\n")
            top_k = 10
            feature_scores = []
            for idx in range(range_feat[0], range_feat[1]):
                if node_mask_cpu[idx]:
                    score = cumulative_scores_cpu[idx].item()
                    selected_idx = idx - range_feat[0]
                    if hasattr(graph, 'active_features') and hasattr(graph, 'selected_features'):
                        layer, pos, feat = graph.active_features[graph.selected_features[selected_idx]].tolist()
                        feature_scores.append((score, layer, pos, feat))
            
            feature_scores.sort(reverse=True)
            for rank, (score, layer, pos, feat) in enumerate(feature_scores[:top_k], 1):
                f.write(f"  {rank}. Feature L{layer} P{pos} F{feat} (score: {score:.4f})\n")
        
        print(f"[VIS] Statistics report saved to: {txt_path}")
        
        if HAS_CIRCUIT_PLOT:
            try:
                print(f"[VIS] Generating simplified circuit diagram (feature nodes only)...")
                graph_stats = visualize_circuit_simple(
                    graph, save_path,
                    node_threshold=node_threshold,
                    edge_threshold=edge_threshold,
                    max_nodes_per_layer=max_nodes_per_layer,
                    show_top_k_edges=show_top_k_edges,
                    tokenizer=tokenizer
                )
                print(f"[VIS] Circuit diagram stats: {graph_stats}")
            except Exception as e:
                print(f"[WARN] Circuit diagram generation failed: {e}")
                import traceback
                traceback.print_exc()
        
        if save_json:
            slug = os.path.basename(base_path)
            json_dir = os.path.dirname(base_path) or "."
            json_dir = os.path.abspath(json_dir)
            
            print(f"[VIS] Creating JSON file for web visualization...")
            try:
                graph.to("cpu")
                create_graph_files(
                    graph_or_path=graph,
                    slug=slug,
                    scan=graph.scan if hasattr(graph, 'scan') else "gemma",
                    output_path=json_dir,
                    node_threshold=node_threshold,
                    edge_threshold=edge_threshold,
                )
                json_file_path = os.path.join(json_dir, f"{slug}.json")
                print(f"[VIS] JSON file saved: {json_file_path}")
                print(f"[VIS] To view in web server, run:")
                print(f"      cd os.path.join(ROOT_DIR, "circuit-tracer", "circuit-tracer")")
                print(f"      export PATH=\"~/.conda/envs/ct-env/bin:$PATH\"")
                print(f"      python -m circuit_tracer start-server --graph_file_dir {json_dir} --port 8041")
                
                return {
                    "success": True,
                    "active_nodes": active_nodes,
                    "active_feature_nodes": active_feature_nodes,
                    "active_error_nodes": active_error_nodes,
                    "active_edges": total_edges,
                    "layer_distribution": layer_distribution,
                    "txt_path": txt_path,
                    "pdf_path": save_path if HAS_CIRCUIT_PLOT else None,
                    "json_path": json_file_path
                }
            except Exception as e:
                print(f"[WARN] JSON creation failed: {e}")
                print(f"[INFO] Statistics report and circuit diagram generated successfully")
        
        return {
            "success": True,
            "active_nodes": active_nodes,
            "active_feature_nodes": active_feature_nodes,
            "active_error_nodes": active_error_nodes,
            "active_edges": total_edges,
            "layer_distribution": layer_distribution,
            "txt_path": txt_path,
            "pdf_path": save_path if HAS_CIRCUIT_PLOT else None,
            "json_path": None
        }
        
    except Exception as e:
        print(f"[ERROR] Visualization failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }
