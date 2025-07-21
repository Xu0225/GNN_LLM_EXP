import os
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

# —— 下面三行替换原来的 PROJECT_ROOT、DATA_DIR 定义 ——
FILE_DIR    = os.path.dirname(__file__)                            # .../GNN_LLM_EXP/data/src
DATA_ROOT   = os.path.abspath(os.path.join(FILE_DIR, '..'))        # .../GNN_LLM_EXP/data
ALERT_CSV   = os.path.join(DATA_ROOT, 'alerts', 'node_alerts.csv')  # .../data/alerts/node_alerts.csv
PROCESSED_DIR = os.path.join(DATA_ROOT, 'processed')               # .../data/processed
os.makedirs(PROCESSED_DIR, exist_ok=True)

# 聚合输出
AGG_CSV     = os.path.join(PROCESSED_DIR, 'node_alerts_agg.csv')
# 嵌入向量输出
EMB_TENSOR  = os.path.join(PROCESSED_DIR, 'node_alert_emb.pt')
ID_TENSOR   = os.path.join(PROCESSED_DIR, 'node_ids.pt')

def aggregate_alerts():
    df = pd.read_csv(ALERT_CSV, dtype={'node_id':str,'alert_text':str})
    agg = df.groupby('node_id')['alert_text']\
            .apply(lambda ts: '；'.join(ts))
    agg.to_csv(AGG_CSV, header=['alert_concat'])
    print("聚合完成 →", AGG_CSV)
    return AGG_CSV

def encode_alerts(agg_csv):
    df = pd.read_csv(agg_csv, index_col=0)
    texts = df['alert_concat'].tolist()
    ids   = df.index.tolist()

    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    embs = encoder.encode(texts, batch_size=64, show_progress_bar=True)

    torch.save(torch.tensor(embs), EMB_TENSOR)
    torch.save(ids, ID_TENSOR)
    print("编码完成 →", EMB_TENSOR, ID_TENSOR)

if __name__ == '__main__':
    agg_csv = aggregate_alerts()
    encode_alerts(agg_csv)
