import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GNNLLMFusion(nn.Module):
    def __init__(self, struct_dim, sem_dim, hid_dim, trans_heads=4):
        super().__init__()
        # 1) 结构编码
        self.conv1 = SAGEConv(struct_dim + sem_dim, hid_dim)
        self.conv2 = SAGEConv(hid_dim, hid_dim)
        # 2) 门控融合（在输入端对结构/语义再融合一次，可选）
        self.gate = nn.Linear((struct_dim + sem_dim), hid_dim)
        # 3) Transformer 解码器层（用于链路预测）
        self.trans = nn.TransformerEncoderLayer(
            d_model=hid_dim*2, nhead=trans_heads, dim_feedforward=hid_dim*2
        )
        # 4) 双线性打分 + MLP
        self.bilinear = nn.Bilinear(hid_dim, hid_dim, hid_dim)
        self.mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim//2),
            nn.ReLU(),
            nn.Linear(hid_dim//2, 1)
        )
        # 5) 节点分类头（用于节点恢复）
        self.node_pred = nn.Sequential(
            nn.Linear(hid_dim, hid_dim//2),
            nn.ReLU(),
            nn.Linear(hid_dim//2, 2)  # 二分类
        )

    def encode(self, x, edge_index):
        # x 已经是 [orig_struct || sem_emb]
        g1 = F.relu(self.conv1(x, edge_index))
        g1 = F.dropout(g1, p=0.3, training=self.training)
        z  = self.conv2(g1, edge_index)
        return z

    def link_decode(self, z, edge_index):
        # 拼接每条边节点表示
        src, dst = edge_index
        h_cat = torch.cat([z[src], z[dst]], dim=1).unsqueeze(0)  # [1, E, 2*hid]
        h_att = self.trans(h_cat).squeeze(0)                    # [E, 2*hid]
        # 双线性 + MLP
        b = self.bilinear(z[src], z[dst])                       # [E, hid]
        return self.mlp(b + h_att[:, :b.size(1)])[:, 0]         # [E]

    def node_decode(self, z):
        return self.node_pred(z)                                 # [N, 2]

    def forward_link(self, data):
        z = self.encode(data.x, data.edge_index)
        pos = self.link_decode(z, data.train_pos_edge_index)
        neg = self.link_decode(z, data.train_neg_edge_index)
        return pos, neg

    def forward_node(self, data):
        z = self.encode(data.x, data.edge_index)
        logits = self.node_decode(z)
        return logits
