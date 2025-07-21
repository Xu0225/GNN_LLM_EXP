# src/visualize_noderec_simple.py

import os
import torch
import networkx as nx
import plotly.graph_objects as go
from model_plus import GNNLLMFusion  # 替换为你的模型类名

# —— 配置 ——
topo       = "Cesnet200511"
FILE_DIR   = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(FILE_DIR, '..'))
graph_file = os.path.join(PROJECT_ROOT, 'topologies',    f'{topo}.graphml')
data_file  = os.path.join(PROJECT_ROOT, 'processed',     f'{topo}_noderec.pt')
emb_file   = os.path.join(PROJECT_ROOT, 'processed',     'node_alert_emb.pt')
model_file = os.path.join(PROJECT_ROOT, 'models',        'noderec_best.pt')

# —— 1. 读取图 & 数据 & 模型 ——
G    = nx.read_graphml(graph_file)
data = torch.load(data_file, map_location='cpu', weights_only=False)

sem_dim    = torch.load(emb_file, map_location='cpu').size(1)
struct_dim = data.x.size(1) - sem_dim

model = GNNLLMFusion(struct_dim, sem_dim, hid_dim=128, heads=4).eval()
model.load_state_dict(torch.load(model_file, map_location='cpu'))

with torch.no_grad():
    logits = model.forward_node(data)
    preds  = logits.argmax(dim=1).cpu().numpy()

# —— 2. 收集实际 & 预测缺失节点 local IDs ——
actual_nodes = []
pred_nodes   = []
for idx, full_id in enumerate(data.node_id_list):
    local = full_id.split('_', 1)[1]
    if data.y[idx].item() == 1 and data.test_mask[idx].item():
        actual_nodes.append(local)
    if preds[idx] == 1 and data.test_mask[idx].item():
        pred_nodes.append(local)

# —— 3. 构造布局 & 边轨迹 ——
pos = nx.spring_layout(G, seed=42)
edge_x, edge_y = [], []
for u, v in G.edges():
    x0, y0 = pos[u]; x1, y1 = pos[v]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]

edge_trace = go.Scatter(
    x=edge_x, y=edge_y, mode='lines',
    line=dict(width=1, color='#888'),
    hoverinfo='none',
    showlegend=False
)

# —— 4. 绘制节点 ——
# 其他节点
covered = set(actual_nodes) | set(pred_nodes)
other_x = [pos[n][0] for n in G.nodes() if n not in covered]
other_y = [pos[n][1] for n in G.nodes() if n not in covered]
other_trace = go.Scatter(
    x=other_x, y=other_y, mode='markers',
    marker=dict(symbol='circle-open', size=8, color='lightgray'),
    hoverinfo='none',
    name='其他节点'
)

# 实际缺失
act_x = [pos[n][0] for n in actual_nodes]
act_y = [pos[n][1] for n in actual_nodes]
actual_trace = go.Scatter(
    x=act_x, y=act_y, mode='markers+text',
    marker=dict(symbol='x', size=14, color='green', line=dict(width=2)),
    text=actual_nodes,
    textposition='top center',
    hoverinfo='text',
    name='实际缺失'
)

# 模型预测
pred_x = [pos[n][0] for n in pred_nodes]
pred_y = [pos[n][1] for n in pred_nodes]
pred_trace = go.Scatter(
    x=pred_x, y=pred_y, mode='markers+text',
    marker=dict(symbol='circle', size=14, color='red', line=dict(width=2)),
    text=pred_nodes,
    textposition='bottom center',
    hoverinfo='text',
    name='模型预测'
)

# —— 5. 画图 ——
fig = go.Figure(data=[
    edge_trace,
    other_trace,
    actual_trace,
    pred_trace
])

fig.update_layout(
    title=f"{topo} 节点恢复 可视化",
    showlegend=True,
    legend=dict(title="节点类型"),
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    height=600,
)

fig.show()
