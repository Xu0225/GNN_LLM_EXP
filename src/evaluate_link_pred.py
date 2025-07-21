# src/evaluate_linkpred.py

import os
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from model import GNNLLMFusion
from torch_geometric.loader import DataLoader

# —— 路径配置 ——
FILE_DIR      = os.path.dirname(__file__)
PROJECT_ROOT  = os.path.abspath(os.path.join(FILE_DIR, '..'))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'processed')
MODEL_PATH    = os.path.join(PROJECT_ROOT, 'models', 'linkpred_best.pt')

# —— 加载模型 ——
# 需与训练时一致的 struct_dim、sem_dim、hid_dim
# 这里自动从任意一个样本推断 struct_dim & sem_dim
sample = torch.load(os.path.join(PROCESSED_DIR, os.listdir(PROCESSED_DIR)[0]), weights_only=False)
sem_dim   = torch.load(os.path.join(PROCESSED_DIR, 'node_alert_emb.pt')).size(1)
struct_dim= sample.x.size(1) - sem_dim
model = GNNLLMFusion(struct_dim, sem_dim, hid_dim=128).eval()
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))

# —— 遍历所有 linkpred 数据集 ——
aucs, aps = [], []
files = sorted([f for f in os.listdir(PROCESSED_DIR) if f.endswith('_linkpred.pt')])
for fname in files:
    data = torch.load(os.path.join(PROCESSED_DIR, fname), weights_only=False)
    # 组合正负测试边
    pos, neg = data.test_pos_edge_index, data.test_neg_edge_index
    edge_index = torch.cat([pos, neg], dim=1)
    # 推理
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index)
        scores = model.link_decode(z, edge_index).cpu().numpy()
        labels = np.concatenate([np.ones(pos.size(1)), np.zeros(neg.size(1))])
        probs  = 1/(1+np.exp(-scores))
    # 计算单图指标
    auc  = roc_auc_score(labels, probs)
    ap   = average_precision_score(labels, probs)
    aucs.append(auc)
    aps.append(ap)

# —— 输出结果和统计 ——
aucs = np.array(aucs); aps = np.array(aps)
print("链路预测评估（共 %d 张图）" % len(aucs))
print(f"平均 AUC: {aucs.mean():.3f}  标准差: {aucs.std():.3f}")
print(f"平均 AP:  {aps.mean():.3f}  标准差: {aps.std():.3f}")
# 如果需要，可输出分布区间
print(f"AUC 分布: min {aucs.min():.3f}, max {aucs.max():.3f}")
print(f"AP  分布: min {aps.min():.3f}, max {aps.max():.3f}")
