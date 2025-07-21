#!/usr/bin/env python3
"""
eval.py

计算原始拓扑与还原拓扑的恢复评测指标，并将结果打印到控制台或保存到 CSV 文件。

使用方法：
1. 将本脚本和 `Chinanet.graphml`、`model_output.json` 放在同一目录。
2. 安装依赖：
     pip install networkx pandas matplotlib
3. 运行：
     python eval.py
"""

import os
import re
import json
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

# —— 配置区 ——
ORIG_GRAPHML    = "./topologies/Chinanet.graphml"
MODEL_JSON      = "model_output.json"
SAVE_CSV        = False        # 是否将指标保存为 CSV
CSV_PATH        = "metrics.csv"

def load_restored_graph(json_file):
    text = open(json_file, "r", encoding="utf-8", errors="ignore").read()
    m = re.search(r'(<\?xml[\s\S]*?</graphml>)', text) or re.search(r'(<graphml[\s\S]*?</graphml>)', text)
    if not m:
        raise RuntimeError("未能提取 GraphML 内容")
    xml = m.group(1)
    nodes = re.findall(r'<node\s+id="([^"]+)"', xml)
    edges = re.findall(r'<edge\s+source="([^"]+)"\s+target="([^"]+)"', xml)
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return G

def compute_metrics(G_orig, G_restored):
    orig_nodes = set(G_orig.nodes())
    rest_nodes = set(G_restored.nodes())
    orig_edges = set(tuple(sorted(e)) for e in G_orig.edges())
    rest_edges = set(tuple(sorted(e)) for e in G_restored.edges())

    # Node metrics
    node_tp = len(orig_nodes & rest_nodes)
    node_fp = len(rest_nodes - orig_nodes)
    node_fn = len(orig_nodes - rest_nodes)
    node_precision = node_tp / (node_tp + node_fp) if node_tp+node_fp else 0
    node_recall    = node_tp / (node_tp + node_fn) if node_tp+node_fn else 0
    node_f1        = (2*node_precision*node_recall/(node_precision+node_recall)
                      if node_precision+node_recall else 0)

    # Edge metrics
    edge_tp = len(orig_edges & rest_edges)
    edge_fp = len(rest_edges - orig_edges)
    edge_fn = len(orig_edges - rest_edges)
    edge_precision = edge_tp/(edge_tp+edge_fp) if edge_tp+edge_fp else 0
    edge_recall    = edge_tp/(edge_tp+edge_fn) if edge_tp+edge_fn else 0
    edge_f1        = (2*edge_precision*edge_recall/(edge_precision+edge_recall)
                      if edge_precision+edge_recall else 0)

    df = pd.DataFrame([
        {
            "Type": "Nodes",
            "Original": len(orig_nodes),
            "Restored": len(rest_nodes),
            "TP": node_tp,
            "FP": node_fp,
            "FN": node_fn,
            "Precision": node_precision,
            "Recall": node_recall,
            "F1": node_f1
        },
        {
            "Type": "Edges",
            "Original": len(orig_edges),
            "Restored": len(rest_edges),
            "TP": edge_tp,
            "FP": edge_fp,
            "FN": edge_fn,
            "Precision": edge_precision,
            "Recall": edge_recall,
            "F1": edge_f1
        }
    ])
    return df

def main():
    if not os.path.isfile(ORIG_GRAPHML):
        print(f"未找到原始文件: {ORIG_GRAPHML}")
        return
    if not os.path.isfile(MODEL_JSON):
        print(f"未找到模型输出文件: {MODEL_JSON}")
        return

    G_orig = nx.read_graphml(ORIG_GRAPHML)
    G_restored = load_restored_graph(MODEL_JSON)

    metrics_df = compute_metrics(G_orig, G_restored)

    # 打印指标
    print("\n=== Topology Recovery Metrics ===")
    print(metrics_df.to_string(index=False, float_format="%.4f"))

    # 可选保存到 CSV
    if SAVE_CSV:
        metrics_df.to_csv(CSV_PATH, index=False)
        print(f"\n已保存指标到: {CSV_PATH}")

    # 可视化布局保持一致
    G_union = nx.Graph()
    G_union.add_nodes_from(set(G_orig.nodes()) | set(G_restored.nodes()))
    pos = nx.spring_layout(G_union, seed=42)

    fig, axes = plt.subplots(1, 2, figsize=(14,7))
    nx.draw(G_orig, pos, ax=axes[0], with_labels=True, node_size=100)
    axes[0].set_title("Original Topology")
    axes[0].axis("off")

    nx.draw(G_restored, pos, ax=axes[1], with_labels=True, node_size=100)
    axes[1].set_title("Restored Topology")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
