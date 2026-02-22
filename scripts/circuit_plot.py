#!/usr/bin/env python3
"""
优化的 Circuit 可视化工具
1. 显示真实 token 文本而不是 Token[N]
2. 只保留有连接的输入节点
3. 更松散的布局，便于查看信息流
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
import sys
import os

# 动态添加 circuit_tracer 路径（相对路径，兼容 LLMvul repo）
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
    """
    生成优化的 circuit 可视化图

    改进：
    1. 显示真实 token 文本（需要传入 tokenizer）
    2. 只保留有出边的输入节点
    3. 更松散的布局

    Args:
        graph: circuit_tracer Graph 对象
        save_path: 保存路径 (PDF)
        node_threshold: 节点修剪阈值
        edge_threshold: 边修剪阈值
        max_nodes_per_layer: 每层最多显示多少个特征节点
        show_top_k_edges: 显示多少条最重要的边
        tokenizer: HuggingFace tokenizer (用于解码token)
    """

    print(f"[VIS] 开始生成优化的 circuit 图...")

    # 确保在正确的设备上
    device = "cuda" if torch.cuda.is_available() else "cpu"
    graph.to(device)

    # 修剪图
    print(f"[VIS] 修剪图 (node_threshold={node_threshold}, edge_threshold={edge_threshold})...")
    node_mask, edge_mask, cumulative_scores = prune_graph(graph, node_threshold, edge_threshold)

    # 移到 CPU
    node_mask = node_mask.cpu()
    edge_mask = edge_mask.cpu()
    cumulative_scores = cumulative_scores.cpu()
    adj = graph.adjacency_matrix.cpu().numpy()

    # 计算范围
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

    # 选择重要的特征节点 (跳过 error 节点)
    print(f"[VIS] 选择重要节点...")
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

    # 按层分组，每层只保留最重要的几个
    nodes_by_layer = defaultdict(list)
    for node_info in important_feature_nodes:
        idx, layer, pos, feat, score = node_info
        nodes_by_layer[layer].append(node_info)

    # 每层排序并限制数量
    selected_nodes = []
    for layer in sorted(nodes_by_layer.keys()):
        layer_nodes = sorted(nodes_by_layer[layer], key=lambda x: x[4], reverse=True)
        selected_nodes.extend(layer_nodes[:max_nodes_per_layer])

    # 临时添加所有嵌入节点（后面会过滤）
    embedding_candidates = []
    for idx in range(range_embed[0], range_embed[1]):
        if node_mask[idx]:
            pos = idx - range_embed[0]
            # 获取 token ID
            if hasattr(graph, 'input_tokens'):
                token_id = graph.input_tokens[pos]
                if hasattr(token_id, 'item'):
                    token_id = token_id.item()
            else:
                token_id = pos

            # 解码真实 token 文本
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

    # 添加输出节点
    for idx in range(range_logit[0], range_logit[1]):
        if node_mask[idx]:
            l_idx = idx - range_logit[0]
            selected_nodes.append((idx, 999, l_idx, 0, cumulative_scores[idx].item()))

    print(f"[VIS] 初步选中 {len(selected_nodes)} 个特征节点, {len(embedding_candidates)} 个候选输入节点")

    # 找到这些节点之间的重要边
    selected_indices = set([n[0] for n in selected_nodes])
    candidate_embedding_indices = set([n[0] for n in embedding_candidates])
    all_candidate_indices = selected_indices | candidate_embedding_indices

    important_edges = []
    for i in all_candidate_indices:
        for j in all_candidate_indices:
            if edge_mask[j, i]:  # j是target, i是source
                weight = adj[j, i]
                if abs(weight) > 1e-6:
                    important_edges.append((i, j, weight))

    # 按权重排序，只保留最重要的边
    important_edges.sort(key=lambda x: abs(x[2]), reverse=True)
    important_edges = important_edges[:show_top_k_edges]

    # 只保留有连接的输入节点
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

    print(f"[VIS] 过滤后保留 {len(filtered_embedding_nodes)} 个有连接的输入节点"
          f"（移除了 {len(embedding_candidates) - len(filtered_embedding_nodes)} 个孤立节点）")

    # 合并所有节点
    selected_nodes.extend(filtered_embedding_nodes)

    print(f"[VIS] 最终选中 {len(selected_nodes)} 个节点, {len(important_edges)} 条边")

    # 创建网络图
    G = nx.DiGraph()

    node_info_dict = {}
    for idx, layer, pos, feat_or_text, score in selected_nodes:
        if layer == -1:  # 输入 token
            label = str(feat_or_text)
            if len(label) > 10:
                label = label[:8] + ".."
            node_type = 'input'
        elif layer == 999:  # 输出 logit
            label = f"Logit[{pos}]"
            node_type = 'output'
        else:  # 特征节点
            label = f"L{layer}\nF{feat_or_text}"
            node_type = 'feature'

        G.add_node(idx, label=label, layer=layer, node_type=node_type, score=score)
        node_info_dict[idx] = (label, layer, node_type)

    # 添加边
    for src, tgt, weight in important_edges:
        if src in G and tgt in G:
            G.add_edge(src, tgt, weight=weight)

    print(f"[VIS] 网络图构建完成: {len(G.nodes())} 节点, {len(G.edges())} 边")

    # 绘制图形
    fig, ax = plt.subplots(figsize=(20, 14))

    # 计算布局
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

    # 绘制边
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

    # 绘制节点
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

    # 标题
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

    # 图例
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

    # 保存 PDF
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[VIS] 图形已保存到: {save_path}")

    # 同时保存 PNG
    png_path = save_path.replace('.pdf', '.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"[VIS] PNG 已保存到: {png_path}")

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
