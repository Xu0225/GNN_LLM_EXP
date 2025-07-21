# src/train_linkpred.py

import os
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from model import GNNLLMFusion

# ———————————— 超参数 ————————————
LR = 1e-3
WEIGHT_DECAY = 1e-5
EPOCHS = 50

# ———————————— 路径配置 ————————————
FILE_DIR      = os.path.dirname(__file__)
PROJECT_ROOT  = os.path.abspath(os.path.join(FILE_DIR, '..'))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'processed')
MODEL_DIR     = os.path.join(PROJECT_ROOT, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# ————————— 加载语义嵌入维度 —————————
sem_emb = torch.load(os.path.join(PROCESSED_DIR, 'node_alert_emb.pt'))
SEM_DIM = sem_emb.size(1)

# —————— 加载链路预测数据集 ——————
files = [f for f in os.listdir(PROCESSED_DIR) if f.endswith('_linkpred.pt')]
dataset = [
    torch.load(os.path.join(PROCESSED_DIR, f), weights_only=False)
    for f in files
]
# 删除 node_id_list，避免多余属性
for data in dataset:
    if hasattr(data, 'node_id_list'):
        del data.node_id_list

# ——————— 输入维度 ——————————
INPUT_DIM  = dataset[0].x.size(1)
STRUCT_DIM = INPUT_DIM - SEM_DIM

# ————————— 模型与优化器 —————————
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = GNNLLMFusion(STRUCT_DIM, SEM_DIM, hid_dim=128).to(device)
opt    = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# ——————————— 损失函数 ———————————
def loss_fn(pos, neg):
    scores = torch.cat([pos, neg], dim=0)
    labels = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)], dim=0)
    return torch.nn.functional.binary_cross_entropy_with_logits(scores, labels)

# ———————— 验证函数 ————————
def eval_auc():
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for data in dataset:
            data = data.to(device)
            pos, neg = model.forward_link(data)
            ys.append(torch.cat([torch.ones_like(pos), torch.zeros_like(neg)]).cpu())
            ps.append(torch.sigmoid(torch.cat([pos, neg])).cpu())
    ys = torch.cat(ys).numpy()
    ps = torch.cat(ps).numpy()
    return roc_auc_score(ys, ps), average_precision_score(ys, ps)

# ——————————— 训练循环 ———————————
for epoch in range(1, EPOCHS+1):
    model.train()
    total_loss = 0.0
    for data in dataset:
        data = data.to(device)
        pos, neg = model.forward_link(data)
        loss = loss_fn(pos, neg)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
    auc, ap = eval_auc()
    print(f"Epoch {epoch:02d} | Loss {total_loss/len(dataset):.4f} | AUC {auc:.4f} | AP {ap:.4f}")

# —————————— 保存模型 ——————————
out_path = os.path.join(MODEL_DIR, 'linkpred_best.pt')
torch.save(model.state_dict(), out_path)
print("链路预测模型已保存至：", out_path)
