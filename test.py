# 快速验证
import pandas as pd, torch
df   = pd.read_csv('./processed/node_alerts_agg.csv', index_col=0)
embs = torch.load('./processed/node_alert_emb.pt')
ids  = torch.load('./processed/node_ids.pt')
print("聚合节点数:", len(df), "嵌入形状:", embs.size())
