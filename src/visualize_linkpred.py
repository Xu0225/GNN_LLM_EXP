# src/visualize_linkpred.py

import os
import torch
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from model import GNNLLMFusion

# —— 配置 —— #
topo     = "Aarnet"       # 拓扑名称
topk     = 20             # TopK 候选链路
save_csv = True           # 是否保存预测结果为 CSV

# 路径
FILE_DIR     = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(FILE_DIR, '..'))
graph_file   = os.path.join(PROJECT_ROOT, 'topologies', f'{topo}.graphml')
data_file    = os.path.join(PROJECT_ROOT, 'processed', f'{topo}_linkpred.pt')
model_file   = os.path.join(PROJECT_ROOT, 'models', 'linkpred_best.pt')

# —— 1. 加载原图、数据、模型 —— #
G = nx.read_graphml(graph_file)
data = torch.load(data_file, weights_only=False)

# 构造节点列表映射：index -> 原始节点 ID
node_list = list(G.nodes())

# 语义 & 结构维度
sem_emb    = torch.load(os.path.join(PROJECT_ROOT, 'processed', 'node_alert_emb.pt'))
sem_dim    = sem_emb.size(1)
struct_dim = data.x.size(1) - sem_dim

model = GNNLLMFusion(struct_dim, sem_dim, hid_dim=128).eval()
state = torch.load(model_file, map_location='cpu')
model.load_state_dict(state)

# —— 2. 推理 TopK 候选边 —— #
edge_index = torch.cat([data.test_pos_edge_index, data.test_neg_edge_index], dim=1)
with torch.no_grad():
    z      = model.encode(data.x, data.edge_index)
    scores = model.link_decode(z, edge_index).cpu().numpy()
    probs  = 1 / (1 + np.exp(-scores))

# 选 TopK
idxs       = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:topk]
pred_edges = [edge_index[:, i].tolist() for i in idxs]

# 用 node_list 映射 u,v 到节点 ID
pred_pairs = [
    (node_list[u], node_list[v], float(probs[i]))
    for (u, v), i in zip(pred_edges, idxs)
]

# —— 3. 输出 & 保存 —— #
df = pd.DataFrame(pred_pairs, columns=['node_u', 'node_v', 'probability'])
print("\n==== TopK 链路预测结果 ====")
print(df.to_string(index=False))

if save_csv:
    out_csv = os.path.join(PROJECT_ROOT, f'{topo}_linkpred_top{topk}.csv')
    df.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f"✅ 已保存至：{out_csv}")

# —— 4. 可视化子图 —— #
focus_nodes = set(df['node_u']).union(df['node_v'])
sub_nodes   = set(focus_nodes)
for nid in focus_nodes:
    sub_nodes.update(G.neighbors(nid))

subG = G.subgraph(sub_nodes).copy()

# 坐标映射（GraphML 中应有 Latitude/Longitude 属性）
pos = {
    n: (
        float(d.get('Longitude', 0)),
        float(d.get('Latitude', 0))
    )
    for n, d in subG.nodes(data=True)
}

plt.figure(figsize=(8,6))
nx.draw(
    subG, pos,
    node_color='lightblue',
    edge_color='gray',
    node_size=100,
    with_labels=True,
    font_size=6
)

# 红色高亮预测边
for u, v, _ in pred_pairs:
    if subG.has_node(u) and subG.has_node(v):
        nx.draw_networkx_edges(
            subG, pos,
            edgelist=[(u, v)],
            edge_color='red',
            width=2
        )

plt.title(f"{topo} 链路预测 Top{topk}（红色高亮）")
plt.axis('off')
plt.tight_layout()
plt.show()
