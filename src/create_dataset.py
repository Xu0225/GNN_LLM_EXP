import os
import random
import torch
import networkx as nx
import numpy as np
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler

# —— 路径配置 ——
FILE_DIR      = os.path.dirname(__file__)
PROJECT_ROOT  = os.path.abspath(os.path.join(FILE_DIR, '..'))
GRAPHML_DIR   = os.path.join(PROJECT_ROOT, 'topologies')
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'processed')
os.makedirs(PROCESSED_DIR, exist_ok=True)

# —— 加载语义嵌入 ——
sem_emb  = torch.load(os.path.join(PROCESSED_DIR, 'node_alert_emb.pt'))
node_ids = torch.load(os.path.join(PROCESSED_DIR, 'node_ids.pt'))
id2idx   = {nid: i for i, nid in enumerate(node_ids)}
EMB_DIM  = sem_emb.size(1)

# —— 保留的结构特征字段 ——
NODE_ATTRS = ['Internal', 'Latitude', 'Longitude']

# —— 缺失节点比例 ——
TRAIN_NODE_RATIO = VAL_NODE_RATIO = TEST_NODE_RATIO = 0.1

def normalize_struct_features(G, attr_keys):
    feats = []
    for _, attrs in G.nodes(data=True):
        vec = [attrs[k] for k in attr_keys]
        feats.append(vec)
    scaler = StandardScaler()
    feats_np = scaler.fit_transform(feats)
    for (nid, attrs), vec in zip(G.nodes(data=True), feats_np):
        for i, k in enumerate(attr_keys):
            attrs[k] = float(vec[i])
    return G

def merge_semantic(data: Data):
    sem_feats = []
    for global_nid in data.node_id_list:
        idx = id2idx.get(global_nid)
        sem_feats.append(sem_emb[idx] if idx is not None else torch.zeros(EMB_DIM))
    sem_feats = torch.stack(sem_feats, dim=0)
    data.x = torch.cat([data.x, sem_feats], dim=1)
    return data

# —— 主循环 ——
for fname in sorted(os.listdir(GRAPHML_DIR)):
    if not fname.endswith('.graphml'):
        continue
    topo = fname[:-8]
    path = os.path.join(GRAPHML_DIR, fname)
    print(f"\n→ 处理拓扑：{topo}")

    # 1. 读取 & 筛选
    G = nx.read_graphml(path)
    if not nx.is_connected(G):
        print("   跳过：非连通图"); continue
    n = G.number_of_nodes()
    if not (5 <= n <= 200):
        print(f"   跳过：节点数 {n} 不在[5,200]"); continue

    # 2. 清除边属性
    for _, _, d in G.edges(data=True):
        d.clear()

    # 3. 强制每个节点有完整结构属性（并转 float）
    for _, attrs in G.nodes(data=True):
        new_attrs = {}
        for k in NODE_ATTRS:
            try:
                val = float(attrs.get(k, 0.0))
            except ValueError:
                val = 0.0
            new_attrs[k] = val
        attrs.clear()
        attrs.update(new_attrs)

    # 4. 标准化结构属性
    G = normalize_struct_features(G, NODE_ATTRS)

    # 5. 转换为 PyG 数据对象
    try:
        data_base = from_networkx(G, group_node_attrs=NODE_ATTRS)
    except Exception as e:
        print(f"   跳过：转换失败 → {e}")
        continue
    data_base.node_id_list = [f"{topo}_{nid}" for nid in G.nodes()]

    # 6. 合并语义向量
    data_base = merge_semantic(data_base)

    # 7. 构造缺失节点采样集
    idxs = list(range(n))
    random.shuffle(idxs)
    n_train = max(1, int(n * TRAIN_NODE_RATIO))
    n_val   = max(1, int(n * VAL_NODE_RATIO))
    n_test  = max(1, int(n * TEST_NODE_RATIO))
    train_n = set(idxs[:n_train])
    val_n   = set(idxs[n_train:n_train+n_val])
    test_n  = set(idxs[n_train+n_val:n_train+n_val+n_test])
    all_n   = train_n | val_n | test_n

    # 8. 替换缺失节点特征为平均值
    mask_vec = data_base.x.mean(dim=0)
    x_full = data_base.x.clone()
    for nid in all_n:
        x_full[nid] = mask_vec

    # 9. 构造标签与掩码
    ei = data_base.edge_index
    mask_edges = [(u not in all_n and v not in all_n)
                  for u, v in zip(ei[0].tolist(), ei[1].tolist())]
    data_nr = Data(
        x = x_full,
        edge_index = ei[:, mask_edges],
        y = torch.tensor([1 if i in all_n else 0 for i in range(n)], dtype=torch.long),
        train_mask = torch.tensor(
            [(i in train_n) or (i not in all_n) for i in range(n)],
            dtype=torch.bool
        ),
        val_mask = torch.tensor([i in val_n for i in range(n)], dtype=torch.bool),
        test_mask= torch.tensor([i in test_n for i in range(n)], dtype=torch.bool)
    )
    data_nr.node_id_list = data_base.node_id_list

    # 10. 保存
    out_nr = os.path.join(PROCESSED_DIR, f"{topo}_noderec.pt")
    torch.save(data_nr, out_nr)
    print(f"   ✔ 保存缺节点还原数据 → {topo}_noderec.pt")

print("\n🎉 所有数据构建完成！")
