# src/preprocess.py

import os
import networkx as nx
import torch
from torch_geometric.utils import from_networkx, train_test_split_edges
from torch_geometric.data import Data

# 原始图所在目录
RAW_DIR = os.path.join(os.path.dirname(__file__), '../data/topologies')
# 处理后数据保存目录
OUT_DIR = os.path.join(os.path.dirname(__file__), '../data/processed')
os.makedirs(OUT_DIR, exist_ok=True)

# 筛选条件
MIN_NODES = 5
MAX_NODES = 200
MIN_EDGES = 3

def process_graph(path):
    # 1. 读取 GraphML
    G = nx.read_graphml(path)
    # 2. 筛选节点数、边数和连通性
    if not nx.is_connected(G):
        return None
    if not (MIN_NODES <= G.number_of_nodes() <= MAX_NODES):
        return None
    if G.number_of_edges() < MIN_EDGES:
        return None

    # 3. 转换成 PyG Data
    data = from_networkx(G)
    # 4. 划分边集：train/val/test = 0.8/0.1/0.1
    data = train_test_split_edges(data, val_ratio=0.1, test_ratio=0.1)
    return data

def main():
    files = [f for f in os.listdir(RAW_DIR) if f.endswith('.graphml')]
    for fname in files:
        path = os.path.join(RAW_DIR, fname)
        data = process_graph(path)
        if data is None:
            print(f"跳过 {fname}：不符合筛选条件")
            continue

        # 保存为 .pt 文件
        out_path = os.path.join(OUT_DIR, fname.replace('.graphml', '.pt'))
        torch.save(data, out_path)
        print(f"已处理并保存：{out_path}")

if __name__ == '__main__':
    main()
