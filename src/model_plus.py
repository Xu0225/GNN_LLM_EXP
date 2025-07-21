# src/model.py

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, GraphNorm

class CrossModalFusion(nn.Module):
    """
    跨模态注意力融合：用结构向量 queries，语义向量 keys/values，
    计算注意力后与结构向量拼接再过一层线性变换。
    """
    def __init__(self, struct_dim, sem_dim, hid_dim):
        super().__init__()
        self.Wq = nn.Linear(struct_dim, hid_dim, bias=False)
        self.Wk = nn.Linear(sem_dim, hid_dim, bias=False)
        self.Wv = nn.Linear(sem_dim, hid_dim, bias=False)
        self.out = nn.Linear(struct_dim + hid_dim, struct_dim)

    def forward(self, h_struct, h_sem):
        # h_struct: [N, struct_dim], h_sem: [N, sem_dim]
        Q = self.Wq(h_struct)                     # [N, hid]
        K = self.Wk(h_sem)                        # [N, hid]
        V = self.Wv(h_sem)                        # [N, hid]
        # 注意力分数：Q·K^T 按行 softmax
        scores = (Q * K).sum(dim=-1, keepdim=True)  # [N,1]
        attn = torch.sigmoid(scores) * V           # [N, hid]
        fused = torch.cat([h_struct, attn], dim=1) # [N, struct_dim+hid]
        return self.out(fused)                     # [N, struct_dim]

class LinkDecoder(nn.Module):
    """
    边预测解码器：对两个节点的 embedding 做双线性打分。
    """
    def __init__(self, hid_dim):
        super().__init__()
        self.bilinear = nn.Bilinear(hid_dim, hid_dim, 1)

    def forward(self, z, edge_index):
        # z: [N, hid_dim], edge_index: [2, E]
        src, dst = edge_index
        return self.bilinear(z[src], z[dst]).squeeze(-1)  # [E]

class GNNLLMFusion(nn.Module):
    """
    优化后的 GNN+LLM 融合模型
    - 两层多头 GAT + GraphNorm + ReLU
    - 跨模态注意力融合模块
    - 节点分类 head & 边预测 head
    """
    def __init__(self, struct_dim, sem_dim, hid_dim=128, heads=4):
        super().__init__()
        self.struct_dim = struct_dim
        self.sem_dim    = sem_dim
        in_dim = struct_dim + sem_dim

        # 融合层：先简单拼接，再跨模态注意融合
        self.fusion = CrossModalFusion(struct_dim, sem_dim, hid_dim)

        # GNN 编码器：两层 GAT + GraphNorm
        self.gat1 = GATConv(in_dim, hid_dim, heads=heads, concat=True)
        self.norm1 = GraphNorm(hid_dim * heads)
        self.gat2 = GATConv(hid_dim * heads, hid_dim, heads=1, concat=False)
        self.norm2 = GraphNorm(hid_dim)

        # Head
        self.node_pred = nn.Linear(hid_dim, 2)
        self.link_pred = LinkDecoder(hid_dim)

    def encode(self, x_struct, x_sem, edge_index):
        # 1) 跨模态融合
        h0 = self.fusion(x_struct, x_sem)  # [N, struct_dim]
        # 2) 拼接语义部分
        h = torch.cat([h0, x_sem], dim=1)  # [N, struct+sem]
        # 3) GAT 层
        h = F.elu(self.norm1(self.gat1(h, edge_index)))
        h = F.elu(self.norm2(self.gat2(h, edge_index)))
        return h  # [N, hid_dim]

    def forward_node(self, data):
        # data.x = [struct || sem], 分离两者
        x = data.x
        x_struct = x[:, :self.struct_dim]
        x_sem    = x[:, self.struct_dim:]
        z = self.encode(x_struct, x_sem, data.edge_index)
        return self.node_pred(z)

    def forward_link(self, data):
        # data: 包含 x 和 edge splits train/val/test
        x = data.x
        x_struct = x[:, :self.struct_dim]
        x_sem    = x[:, self.struct_dim:]
        z = self.encode(x_struct, x_sem, data.edge_index)

        pos = self.link_pred(z, data.train_pos_edge_index)
        neg = self.link_pred(z, data.train_neg_edge_index)
        return pos, neg
