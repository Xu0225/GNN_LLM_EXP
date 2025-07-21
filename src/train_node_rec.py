# src/train_noderec.py

import os
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from model import GNNLLMFusion
from model_plus import GNNLLMFusion
# ———————————— 超参数 ————————————
BATCH_SIZE   = 16
LR            = 1e-3
WEIGHT_DECAY  = 1e-5
EPOCHS        = 50

# ———————————— 路径配置 ————————————
FILE_DIR      = os.path.dirname(__file__)
PROJECT_ROOT  = os.path.abspath(os.path.join(FILE_DIR, '..'))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'processed')
MODEL_DIR     = os.path.join(PROJECT_ROOT, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# ———————————— 加载告警嵌入维度 ————————————
sem_emb = torch.load(os.path.join(PROCESSED_DIR, 'node_alert_emb.pt'), weights_only=False)
SEM_DIM = sem_emb.size(1)

# ———————————— 加载数据集 ————————————
files = sorted([f for f in os.listdir(PROCESSED_DIR) if f.endswith('_noderec.pt')])
dataset = [
    torch.load(os.path.join(PROCESSED_DIR, f), weights_only=False)
    for f in files
]
# 删除非张量属性，避免批处理报错
for data in dataset:
    if hasattr(data, 'node_id_list'):
        del data.node_id_list

train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# ———————————— 计算类不平衡权重 ————————————
total_pos = total_neg = 0
for data in dataset:
    y_train = data.y[data.train_mask]
    total_pos += (y_train == 1).sum().item()
    total_neg += (y_train == 0).sum().item()
pos_weight = total_neg / total_pos if total_pos > 0 else 1.0

# ———————————— 构建模型 ————————————
# 自动推断输入维度
example   = dataset[0]
INPUT_DIM = example.x.size(1)
STRUCT_DIM= INPUT_DIM - SEM_DIM

device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model     = GNNLLMFusion(STRUCT_DIM, SEM_DIM, hid_dim=128).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_weight], device=device))

# ———————————— 训练 & 验证 循环 ————————————
for epoch in range(1, EPOCHS+1):
    # 训练
    model.train()
    train_loss = 0.0
    for data in train_loader:
        data   = data.to(device)
        logits = model.forward_node(data)
        loss   = criterion(logits[data.train_mask], data.y[data.train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # 验证
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for data in val_loader:
            data   = data.to(device)
            logits = model.forward_node(data)
            preds  = logits.argmax(dim=1)
            vm     = data.val_mask
            all_preds.append(preds[vm].cpu())
            all_labels.append(data.y[vm].cpu())
    if all_labels:
        all_preds  = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        val_acc  = accuracy_score(all_labels, all_preds)
        val_prec = precision_score(all_labels, all_preds, zero_division=0)
        val_rec  = recall_score(all_labels, all_preds, zero_division=0)
        val_f1   = f1_score(all_labels, all_preds, zero_division=0)
    else:
        val_acc = val_prec = val_rec = val_f1 = 0.0

    print(f"Epoch {epoch:02d} | Train Loss {train_loss:.4f} | "
          f"Val Acc {val_acc:.4f} | Val Prec {val_prec:.4f} | "
          f"Val Rec {val_rec:.4f} | Val F1 {val_f1:.4f}")

# ———————————— 保存模型 ————————————
out_path = os.path.join(MODEL_DIR, 'noderec_best.pt')
torch.save(model.state_dict(), out_path)
print("节点还原模型已保存至：", out_path)
