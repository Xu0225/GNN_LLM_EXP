# src/visualize_noderec_plotly.py

import os
import torch
import networkx as nx
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from model_plus import GNNLLMFusion  # 请替换为你实际的模型类名

# —— 配置路径 ——
topo       = "Aarnet"
FILE_DIR   = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(FILE_DIR, '..'))
graph_file = os.path.join(PROJECT_ROOT, 'topologies',       f'{topo}.graphml')
data_file  = os.path.join(PROJECT_ROOT, 'processed',        f'{topo}_noderec.pt')
emb_file   = os.path.join(PROJECT_ROOT, 'processed',        'node_alert_emb.pt')
model_file = os.path.join(PROJECT_ROOT, 'models',           'noderec_best.pt')
alerts_csv = os.path.join(PROJECT_ROOT, 'alerts',           'node_alerts.csv')

# —— 1. 读取并过滤告警 ——
df_alerts = pd.read_csv(alerts_csv, dtype={'node_id':str, 'alert_text':str})
# 只保留本拓扑关联的 local_id
# data.node_id_list 中保存的是 "topo_local" 格式
data = torch.load(data_file, map_location='cpu', weights_only=False)
local_ids = {nid.split('_', 1)[1] for nid in data.node_id_list}
df_alerts = df_alerts[df_alerts['node_id'].isin(local_ids)]

# 按 local_id 聚合告警文本，用 HTML <br> 分行
alerts = (
    df_alerts
    .groupby('node_id')['alert_text']
    .apply(lambda seq: '<br>'.join(seq))
    .to_dict()
)

# —— 2. 读取图与模型推理 ——
G        = nx.read_graphml(graph_file)
sem_dim  = torch.load(emb_file, map_location='cpu').size(1)
struct_dim = data.x.size(1) - sem_dim

model = GNNLLMFusion(struct_dim, sem_dim, hid_dim=128, heads=4).eval()
model.load_state_dict(torch.load(model_file, map_location='cpu'))
with torch.no_grad():
    logits = model.forward_node(data)
    preds  = logits.argmax(dim=1).cpu().numpy()

# —— 3. 准备节点信息 ——
node_info = []
for idx, full_id in enumerate(data.node_id_list):
    _, local = full_id.split('_', 1)
    is_actual = (data.y[idx].item() == 1 and data.test_mask[idx].item())
    is_pred   = (preds[idx] == 1 and data.test_mask[idx].item())
    text      = alerts.get(local, "无告警信息")
    node_info.append({
        'full_id':       full_id,
        'local':         local,
        'actual_missing': is_actual,
        'pred_missing':   is_pred,
        'alert':          text
    })

# —— 4. 布局 & 边线轨迹 ——
pos = nx.spring_layout(G, seed=42)
edge_x, edge_y = [], []
for u, v in G.edges():
    x0, y0 = pos[u]; x1, y1 = pos[v]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]

edge_trace = go.Scatter(
    x=edge_x, y=edge_y, mode='lines',
    line=dict(width=1, color='#888'),
    hoverinfo='none'
)

# —— 5. 节点轨迹 ——
node_traces = []
for label, color, symbol in [
    ('actual_missing', 'green', 'x'),
    ('pred_missing',   'red',   'circle')
]:
    xs, ys, texts = [], [], []
    for info in node_info:
        if not info[label]:
            continue
        x, y = pos[info['local']]
        xs.append(x)
        ys.append(y)
        texts.append(f"{info['full_id']}<br>{info['alert']}")
    node_traces.append(go.Scatter(
        x=xs, y=ys, mode='markers',
        marker=dict(symbol=symbol, size=14, color=color, line=dict(width=2)),
        text=texts, hoverinfo='text', hoverlabel=dict(align='left'),
        name={'actual_missing':'实际缺失','pred_missing':'模型预测'}[label]
    ))

# —— 6. 其它节点 ——
covered = {info['local'] for info in node_info if info['actual_missing'] or info['pred_missing']}
other_x, other_y = [], []
for n in G.nodes():
    if n not in covered:
        x, y = pos[n]
        other_x.append(x)
        other_y.append(y)

other_trace = go.Scatter(
    x=other_x, y=other_y, mode='markers',
    marker=dict(symbol='circle-open', size=8, color='lightgray'),
    hoverinfo='none', name='其他节点'
)

# —— 7. 子图布局：左 network，右 table ——
fig = make_subplots(
    rows=1, cols=2,
    column_widths=[0.65, 0.35],
    specs=[[{"type":"scatter"}, {"type":"table"}]],
    horizontal_spacing=0.02
)

# 添加网络图
fig.add_trace(edge_trace, row=1, col=1)
for tr in node_traces:
    fig.add_trace(tr, row=1, col=1)
fig.add_trace(other_trace, row=1, col=1)

# 右侧告警表格：只列实际/预测缺失节点
alert_nodes = [info for info in node_info if info['actual_missing'] or info['pred_missing']]
table_id    = [info['full_id'] for info in alert_nodes]
table_alert = [info['alert'].replace('<br>','\n') for info in alert_nodes]

fig.add_trace(go.Table(
    header=dict(values=["Node ID", "告警信息"], align="left", fill_color="#fafafa"),
    cells=dict(values=[table_id, table_alert], align="left"),
    columnwidth=[0.3, 0.7]
), row=1, col=2)

# —— 8. 美化并展示 ——
fig.update_layout(
    title=f"{topo} 拓扑缺失节点恢复示例（仅显示关联告警）",
    showlegend=True,
    legend=dict(title="节点类别"),
    margin=dict(l=10, r=10, t=50, b=10),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    height=600
)

fig.show()
