<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:00C9A7,45:2563EB,100:7C3AED&height=230&section=header&text=GNN%20%C3%97%20LLM%20for%20Network%20Intelligence&fontSize=42&fontColor=FFFFFF&fontAlignY=35&desc=Structure-aware%20Graph%20Learning%20%2B%20Semantic%20Reasoning%20for%20Topology%20Recovery&descAlignY=55&descSize=16&animation=fadeIn"/>

# 🌐 GNN × LLM EXP

### 面向网络拓扑预测、恢复与推理的图智能实验框架

<p>
一个结合 <b>图神经网络（GNN）</b>、<b>告警语义</b> 与 <b>大语言模型（LLM）</b> 的研究原型，
用于理解、预测与恢复复杂通信网络。
</p>

<p>
<b>网络拓扑</b> · <b>链路预测</b> · <b>节点恢复</b> · <b>告警语义</b> · <b>LLM 拓扑重建</b>
</p>

<p>
<a href="./README_EN.md">English</a> · <b>中文</b>
</p>

<p>
<img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white"/>
<img src="https://img.shields.io/badge/PyG-Graph%20Neural%20Networks-3B82F6"/>
<img src="https://img.shields.io/badge/NetworkX-GraphML-10B981"/>
<img src="https://img.shields.io/badge/MiniLM-Alert%20Embeddings-F59E0B"/>
<img src="https://img.shields.io/badge/LLM-Topology%20Reasoning-7C3AED"/>
<img src="https://img.shields.io/github/stars/Xu0225/GNN_LLM_EXP?style=flat&logo=github"/>
</p>

<br/>

**图结构 + 告警语义 → GNN 表征学习 → 网络预测 → LLM 推理 → 拓扑恢复**

</div>

---

## ⚡ 项目简介

现代通信网络天然适合表示为图：

- **设备** 是节点；
- **物理 / 逻辑连接** 是边；
- **位置与角色属性** 描述网络结构；
- **告警与运维事件** 提供运行语义。

这个项目尝试回答一个很直接的问题：

> **当网络拓扑出现缺失或异常时，图结构学习与语言模型推理能否互补？**

**GNN_LLM_EXP** 实现了两条相互补充的智能分析路径：

| 引擎 | 学习 / 推理对象 | 主要任务 |
|---|---|---|
| 🧠 **GNN 引擎** | 图结构 + 告警语义向量 | 链路预测、缺失节点检测 / 恢复 |
| ✨ **LLM 引擎** | 拓扑描述 + 运维告警 + 场景知识 | 拓扑重建与推理 |

项目覆盖了完整实验链路：

**拓扑预处理 → 告警模拟 → 语义编码 → 图学习 → 推理 → 可视化 → LLM Prompt 生成 → 拓扑恢复评测**

---

# 🧬 系统架构

~~~mermaid
flowchart LR
    T["🌐 GraphML 拓扑<br/>节点 · 链路 · 地理属性"]
    A["🚨 网络告警<br/>CPU · 时延 · 接口 · 流量"]

    A --> AE["📝 告警聚合"]
    AE --> SE["🔤 Sentence Transformer<br/>all-MiniLM-L6-v2"]
    SE --> SEM["语义向量"]

    T --> SF["📐 结构特征<br/>Internal · Latitude · Longitude"]
    SF --> FUSE["⚡ 结构 + 语义融合"]
    SEM --> FUSE

    FUSE --> GNN["🧠 GNN 编码器<br/>GraphSAGE / GAT"]
    GNN --> LP["🔗 链路预测"]
    GNN --> NR["🧩 缺失节点恢复"]

    T --> SCENE["🎭 场景化映射<br/>Wireless / Transport"]
    SCENE --> MASK["💥 拓扑破坏<br/>删除节点 / 边"]
    MASK --> PROMPT["📜 Prompt 构建<br/>拓扑 + 告警 + Few-shot"]
    PROMPT --> LLM["✨ 大语言模型"]
    LLM --> RESTORE["🔧 恢复后的 GraphML"]

    LP --> VIZ["📊 评测与可视化"]
    NR --> VIZ
    RESTORE --> EV["✅ 节点 / 边恢复指标"]

    style GNN fill:#2563EB,color:#fff
    style LLM fill:#7C3AED,color:#fff
    style FUSE fill:#00A98F,color:#fff
~~~

---

# 🔥 核心设计

<table>
<tr>
<td width="50%" valign="top">

## 🧠 结构—语义融合

项目并不只根据节点连接关系来表示一个网络设备。

每个节点同时结合：

- Internal / External 标记
- Latitude
- Longitude

以及来自运维告警的语义表示。

节点告警先聚合，再通过：

`all-MiniLM-L6-v2`

编码为语义向量。

因此，每个图节点既包含：

> **它在哪里、如何连接**

也包含：

> **它正在经历什么**

</td>
<td width="50%" valign="top">

## ✨ LLM 拓扑推理

LLM 分支会把抽象拓扑映射成更贴近真实通信网络的场景。

**无线场景**

`BBU · RRU · SW`

**传输场景**

`OLT · ONU · SW`

随后主动删除部分节点或链路，让 LLM 接收：

- 剩余拓扑结构
- 近期告警
- Few-shot 示例
- 严格结构化输出格式

最终目标是恢复缺失的网络结构。

</td>
</tr>

<tr>
<td width="50%" valign="top">

## 🔗 链路预测

对于不完整图，GNN 判断两个节点之间是否应存在连接。

基线模型使用：

- GraphSAGE 编码
- Transformer 边交互
- 双线性打分
- MLP 预测头

评测指标：

`ROC-AUC` · `Average Precision`

</td>
<td width="50%" valign="top">

## 🧩 缺失节点恢复

增强模型使用：

- Cross-Modal Fusion
- Multi-head Graph Attention
- GraphNorm
- 节点分类头

训练阶段主动遮蔽部分节点，让模型利用拓扑和语义上下文识别潜在缺失节点。

评测：

`Accuracy` · `Precision` · `Recall` · `F1`

</td>
</tr>
</table>

---

# 🧠 GNN 模型设计

## 基线模型 — GraphSAGE + Transformer

`src/model.py`

~~~mermaid
flowchart LR
    X["节点特征<br/>结构 + 语义"]
    X --> S1["GraphSAGE"]
    S1 --> S2["GraphSAGE"]
    S2 --> Z["节点表示"]

    Z --> C["节点对拼接"]
    C --> TR["Transformer Encoder"]
    Z --> BI["Bilinear Interaction"]

    TR --> M["融合"]
    BI --> M
    M --> P["链路概率"]

    Z --> N["节点分类头"]
    N --> R["缺失 / 正常"]
~~~

模型提供两个任务头：

~~~text
forward_link()  → 正 / 负边打分
forward_node()  → 节点二分类 logits
~~~

---

## 增强模型 — Cross-Modal Attention + GAT

`src/model_plus.py`

增强模型显式区分：

~~~text
结构特征  ×  语义特征
~~~

并在图传播前先进行跨模态融合。

~~~mermaid
flowchart LR
    ST["📐 结构向量"]
    SM["💬 告警语义向量"]

    ST --> Q["Query Projection"]
    SM --> KV["Key / Value Projection"]

    Q --> ATT["Cross-Modal Attention"]
    KV --> ATT

    ATT --> CAT["特征融合"]
    SM --> CAT

    CAT --> G1["Multi-head GAT"]
    G1 --> GN1["GraphNorm + ELU"]
    GN1 --> G2["GAT"]
    G2 --> GN2["GraphNorm + ELU"]
    GN2 --> Z["网络感知表示"]

    Z --> NODE["节点恢复头"]
    Z --> LINK["双线性链路解码器"]
~~~

这一结构体现了项目最核心的思想：

> **拓扑告诉模型“节点之间是什么关系”，告警语义告诉模型“节点当前发生了什么”。**

---

# 🚨 告警智能

网络告警由 `src/simulate_alerts.py` 生成。

当前模拟事件包括：

| 告警类型 | 含义 |
|---|---|
| `interface_down` | 接口或端口异常 |
| `link_latency` | 链路往返时延异常 |
| `traffic_spike` | 带宽占用突增 |
| `cpu_high` | CPU 过载 |
| `memory_full` | 内存压力 |

生成后的告警保存在：

~~~text
alerts/node_alerts.csv
~~~

随后 `src/encode.py` 会：

1. 聚合同一节点的多条告警；
2. 使用 **SentenceTransformer** 对告警文本编码；
3. 将语义向量保存到处理后的数据集中。

输出：

~~~text
processed/node_alerts_agg.csv
processed/node_alert_emb.pt
processed/node_ids.pt
~~~

---

# 🌐 拓扑数据集

仓库中包含多种真实网络风格的 **GraphML** 拓扑。

例如：

~~~text
Aarnet
Abilene
Aconet
Agis
Airtel
Arpanet196912
Arpanet19706
Arpanet19719
...
~~~

每个拓扑都作为一个独立图样本参与实验。

数据构建阶段主要保留节点数满足：

~~~text
5 ≤ 节点数 ≤ 200
~~~

的连通图，并对结构属性标准化，再与告警语义特征进行融合。

---

# 🚀 快速开始

## 1. 克隆

~~~bash
git clone https://github.com/Xu0225/GNN_LLM_EXP.git
cd GNN_LLM_EXP
~~~

## 2. 安装环境

~~~bash
pip install -r requirements.txt
~~~

环境中包含：

~~~text
PyTorch
PyTorch Geometric
NetworkX
Sentence Transformers
Plotly
scikit-learn
LLM 相关实验依赖
~~~

> 当前 `requirements.txt` 是原始开发环境的完整快照，其中包含 GPU / CUDA 相关 PyTorch 包。  
> 如果在不同 CUDA 或纯 CPU 环境运行，建议先安装与你本机匹配的 PyTorch 版本。

---

# 🧪 GNN 实验流程

## 第 1 步 — 模拟网络告警

~~~bash
python src/simulate_alerts.py
~~~

输出：

~~~text
alerts/node_alerts.csv
~~~

---

## 第 2 步 — 编码告警语义

~~~bash
python src/encode.py
~~~

使用：

~~~text
SentenceTransformer("all-MiniLM-L6-v2")
~~~

将告警文本转换为节点级稠密语义向量。

---

## 第 3 步 — 构建图数据

~~~bash
python src/create_dataset.py
~~~

处理流程：

~~~text
GraphML
   ↓
筛选 / 标准化
   ↓
节点结构属性
   +
告警语义向量
   ↓
PyTorch Geometric Data
   ↓
节点遮蔽 / 恢复标签
~~~

处理结果保存在：

~~~text
processed/
~~~

仓库中也保留了用于链路预测实验的 `*_linkpred.pt` 数据。

---

## 第 4 步 — 训练链路预测模型

~~~bash
python src/train_link_pred.py
~~~

训练目标：

~~~text
真实链路 → 1
负采样链路 → 0
~~~

评测指标：

~~~text
ROC-AUC
Average Precision
~~~

模型输出：

~~~text
models/linkpred_best.pt
~~~

---

## 第 5 步 — 训练缺失节点恢复模型

~~~bash
python src/train_node_rec.py
~~~

验证指标：

~~~text
Accuracy
Precision
Recall
F1
~~~

模型输出：

~~~text
models/noderec_best.pt
~~~

两个训练好的模型权重都已保存在仓库中。

---

## 第 6 步 — 可视化节点恢复

~~~bash
python src/infer.py
~~~

推理脚本会对比：

- 实际缺失节点
- 模型预测缺失节点
- 其他正常节点

并通过 Plotly 生成交互式网络图。

> 当前 `src/infer.py` 通过脚本开头的 `topo` 变量选择要测试的网络拓扑。

---

# ✨ LLM 拓扑恢复

LLM 实验位于：

~~~text
llm_test/
~~~

这一分支没有绑定固定模型供应商或特定商业 API，而是聚焦于：

**Prompt 构建、拓扑破坏、结构化恢复和结果评测。**

---

## 1. 将抽象拓扑转换为真实网络场景

~~~bash
python llm_test/real_data_proc/trans_to_realsenerio.py   --input topologies/Abilene.graphml   --scenario wireless   --internal_only   --drop_orig_label   --seed 42   --output llm_test/real_data_proc/wireless_scenario.graphml
~~~

支持场景：

| 场景 | 设备命名 |
|---|---|
| **Wireless** | BBU · RRU · SW |
| **Transport** | OLT · ONU · SW |

---

## 2. 生成残缺拓扑与推理 Prompt

~~~bash
python llm_test/real_data_proc/generate_prompt.py   --gml llm_test/real_data_proc/wireless_scenario.graphml   --remove_ratio 0.15   --mode node   --scenario wireless   --with_alerts   --alert_mode aligned   --out llm_test/real_data_proc/wireless_prompt.txt   --mask_out llm_test/real_data_proc/wireless_mask.graphml   --gt_json llm_test/real_data_proc/wireless_gt.json   --seed 42
~~~

会生成三个核心实验文件：

~~~text
wireless_prompt.txt      → 输入给 LLM 的 Prompt
wireless_mask.graphml    → 被破坏后的残缺拓扑
wireless_gt.json         → 隐藏 Ground Truth
~~~

Prompt 要求模型输出：

~~~json
{
  "removed_nodes": [],
  "removed_edges": [],
  "full_graphml": "... reconstructed GraphML ..."
}
~~~

---

## 3. 评测恢复后的拓扑

将 LLM 输出保存到：

~~~text
llm_test/model_output.json
~~~

然后运行：

~~~bash
cd llm_test
python eval.py
~~~

评测会分别比较节点和边：

| 层级 | 指标 |
|---|---|
| **节点** | Precision · Recall · F1 |
| **边** | Precision · Recall · F1 |

同时会并排可视化：

**原始拓扑 vs. 恢复拓扑**

---

# 🔄 两种智能范式

~~~mermaid
flowchart TB
    P["不完整 / 异常网络"]

    P --> G["🧠 GNN 路径"]
    P --> L["✨ LLM 路径"]

    G --> G1["学习图潜在表示"]
    G1 --> G2["预测链路 / 缺失节点"]
    G2 --> G3["统计学习指标评测"]

    L --> L1["将拓扑转成语言上下文"]
    L1 --> L2["基于拓扑 + 告警推理"]
    L2 --> L3["生成恢复后的 GraphML"]

    G3 --> C["🔬 对比两种范式"]
    L3 --> C
~~~

这个项目真正关心的问题并不是简单的：

> **“GNN 和 LLM 谁更好？”**

更值得研究的是：

> **哪些信息应该由图模型从结构中学习，哪些信息应该由语言模型从运维语义中推理，以及两者应该在哪里结合？**

---

# 📊 评测矩阵

| 任务 | 输入 | 模型 | 主要指标 |
|---|---|---|---|
| 🔗 链路预测 | 拓扑 + 语义特征 | GraphSAGE 融合模型 | ROC-AUC, AP |
| 🧩 节点恢复 | 遮蔽图 + 语义特征 | Cross-modal GAT | Accuracy, Precision, Recall, F1 |
| ✨ LLM 恢复 | 残缺拓扑 + 告警 | 外部 LLM | 节点 / 边 Precision, Recall, F1 |
| 📈 可视化 | 图 + 预测结果 | NetworkX / Plotly | 拓扑定性分析 |

---

# 📂 项目结构

~~~text
GNN_LLM_EXP/
│
├── src/
│   ├── simulate_alerts.py        # 模拟运维告警
│   ├── encode.py                 # 告警聚合 + MiniLM 编码
│   ├── create_dataset.py         # 构建 PyG 图数据
│   ├── model.py                  # GraphSAGE + Transformer
│   ├── model_plus.py             # Cross-modal Attention + GAT
│   ├── train_link_pred.py        # 链路预测训练
│   ├── train_node_rec.py         # 缺失节点恢复训练
│   ├── evaluate_link_pred.py
│   ├── evaluate_noderec.py
│   ├── infer.py
│   ├── visualize_linkpred.py
│   └── visualize_noderec.py
│
├── llm_test/
│   ├── generate_prompt.py
│   ├── eval.py
│   ├── eval_graphml.py
│   ├── model_output.json
│   └── real_data_proc/
│       ├── trans_to_realsenerio.py
│       ├── generate_prompt.py
│       ├── wireless_prompt.txt
│       ├── wireless_mask.graphml
│       └── wireless_gt.json
│
├── topologies/                  # GraphML 网络拓扑
├── alerts/                      # 节点级告警
├── processed/                   # PyG 数据与语义嵌入
├── models/
│   ├── linkpred_best.pt
│   └── noderec_best.pt
│
├── Aarnet_linkpred_top20.csv
├── requirements.txt
├── readme.md
└── README_EN.md
~~~

---

# 🛠️ 技术栈

<div align="center">

| 层级 | 技术 |
|---|---|
| 图学习 | **PyTorch · PyTorch Geometric** |
| GNN 算子 | **GraphSAGE · GAT · GraphNorm** |
| 语义编码 | **Sentence Transformers · MiniLM** |
| 图处理 | **NetworkX · GraphML** |
| 评测指标 | **scikit-learn** |
| 可视化 | **Plotly · Matplotlib** |
| LLM 实验 | **Prompt Engineering · Structured JSON · GraphML Reconstruction** |

</div>

---

# 🔬 可扩展研究方向

### 1. GNN → LLM 上下文

将 GNN 的高置信度链路预测和节点恢复结果作为结构化证据提供给 LLM，而不是让 LLM 仅凭原始拓扑推理。

### 2. LLM → GNN 语义增强

使用 LLM 将告警、工单、维护日志和设备描述转化为更丰富的节点语义特征。

### 3. 图—语言联合推理

进一步引入 Graph-aware LLM、GraphRAG 或图检索增强生成机制，减少 GNN 与 LLM 之间的割裂。

### 4. 面向真实网络运维

将模拟告警扩展到：

~~~text
Telemetry · Logs · SNMP · 性能指标 · 工单 · 故障事件
~~~

评测真实故障场景下的拓扑恢复能力。

---

# ⚠️ 实验说明

本仓库是研究原型，而不是已经产品化的 Python 库。

当前仍保留部分原始实验开发习惯：

- 某些推理参数直接在脚本内设置；
- 同时保留了 baseline 与增强版 GNN；
- 为复现实验，仓库中保存了处理后的 `.pt` 数据和模型权重；
- LLM 分支没有硬编码模型 API；
- `requirements.txt` 是完整环境快照，比最小运行依赖更大。

这些设计保留了原始实验过程，也方便进一步阅读和扩展。

---

<div align="center">

## 🌐 从拓扑走向网络智能

**结构告诉我们网络如何连接。**  
**告警告诉我们网络正在经历什么。**  
**图学习与语言推理，把两者连接起来。**

<br/>

⭐ 如果这个项目对你的研究有帮助，欢迎点一个 Star。

<br/>

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:7C3AED,50:2563EB,100:00C9A7&height=130&section=footer"/>

</div>
