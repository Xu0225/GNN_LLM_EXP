# src/evaluate_noderec.py

import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from model import GNNLLMFusion
from torch_geometric.loader import DataLoader
from model_plus import GNNLLMFusion

# —— 路径配置 ——
FILE_DIR      = os.path.dirname(__file__)
PROJECT_ROOT  = os.path.abspath(os.path.join(FILE_DIR, '..'))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'processed')
MODEL_PATH    = os.path.join(PROJECT_ROOT, 'models', 'noderec_best.pt')

# —— 加载模型 ——
# 自动推断 struct_dim & sem_dim
sample     = torch.load(os.path.join(PROCESSED_DIR, os.listdir(PROCESSED_DIR)[0]), weights_only=False)
sem_dim    = torch.load(os.path.join(PROCESSED_DIR, 'node_alert_emb.pt')).size(1)
struct_dim = sample.x.size(1) - sem_dim
model = GNNLLMFusion(struct_dim, sem_dim, hid_dim=128).eval()
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))

# —— 遍历所有 noderec 数据集 ——
accs, precs, recs, f1s = [], [], [], []
files = sorted([f for f in os.listdir(PROCESSED_DIR) if f.endswith('_noderec.pt')])
for fname in files:
    data = torch.load(os.path.join(PROCESSED_DIR, fname), weights_only=False)
    with torch.no_grad():
        logits = model.forward_node(data)
        preds  = logits.argmax(dim=1).cpu().numpy()
        labels = data.y.cpu().numpy()
        # 仅计算 test_mask 节点
        mask   = data.test_mask.cpu().numpy()
        if mask.sum() == 0:
            continue
        y_true = labels[mask]; y_pred = preds[mask]
    # 计算指标
    accs.append   (accuracy_score   (y_true, y_pred))
    precs.append  (precision_score  (y_true, y_pred, zero_division=0))
    recs.append   (recall_score     (y_true, y_pred, zero_division=0))
    f1s.append    (f1_score         (y_true, y_pred, zero_division=0))

# —— 输出结果和统计 ——
accs = np.array(accs); precs = np.array(precs)
recs = np.array(recs); f1s   = np.array(f1s)
print("缺节点还原评估（共 %d 张图）" % len(accs))
print(f"平均 Acc:       {accs.mean():.3f}  标准差: {accs.std():.3f}")
print(f"平均 Precision: {precs.mean():.3f}  标准差: {precs.std():.3f}")
print(f"平均 Recall:    {recs.mean():.3f}  标准差: {recs.std():.3f}")
print(f"平均 F1:        {f1s.mean():.3f}  标准差: {f1s.std():.3f}")
