#!/usr/bin/env python3
"""
vis_from_raw.py

直接从包含 full_graphml 的原始文本（非严格 JSON）中抽取 GraphML，
再用正则解析节点/边并可视化。

用法：
1. 确保当前目录下有 model_output.json（内容即为大模型输出）。
2. 保存本脚本为 vis_from_raw.py。
3. 安装依赖：pip install networkx matplotlib
4. 运行：python vis_from_raw.py
"""

import os
import re
import networkx as nx
import matplotlib.pyplot as plt

# —— 配置区 ——
RAW_FILE = "model_output.json"  # 原始大模型输出文本

def extract_graphml(raw_text):
    """用正则从原始文本中提取 GraphML 区块（<?xml...?</graphml>）。"""
    m = re.search(r'(<\?xml[\s\S]*?</graphml>)', raw_text)
    if not m:
        raise RuntimeError("未能匹配到 <?xml … </graphml> 区块，请检查输入")
    return m.group(1)

def parse_graphml(xml):
    """从 GraphML 文本中提取节点 ID 列表和边列表。"""
    nodes = re.findall(r'<node\s+id="([^"]+)"\s*/?>', xml)
    edges = re.findall(r'<edge\s+source="([^"]+)"\s+target="([^"]+)"\s*/?>', xml)
    return nodes, edges

def build_and_draw(nodes, edges):
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, with_labels=True, node_size=100)
    plt.title("Restored Topology")
    plt.axis("off")
    plt.show()

def main():
    if not os.path.isfile(RAW_FILE):
        print(f"未找到文件: {RAW_FILE}")
        return

    text = open(RAW_FILE, 'r', encoding='utf-8', errors='ignore').read()
    try:
        xml = extract_graphml(text)
    except Exception as e:
        print("提取 GraphML 失败:", e)
        return

    nodes, edges = parse_graphml(xml)
    if not nodes:
        print("未解析到任何节点，请检查 XML 格式")
        return

    build_and_draw(nodes, edges)

if __name__ == "__main__":
    main()
