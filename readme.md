<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:00C9A7,45:2563EB,100:7C3AED&height=230&section=header&text=GNN%20%C3%97%20LLM%20for%20Network%20Intelligence&fontSize=42&fontColor=FFFFFF&fontAlignY=35&desc=Graph%20Learning%20%2B%20Semantic%20Reasoning%20for%20Topology%20Recovery&descAlignY=55&descSize=16&animation=fadeIn"/>

# 🌐 GNN × LLM EXP

### 面向网络拓扑预测、恢复与推理的图智能实验平台

<p>
将 <b>图神经网络（GNN）</b>、<b>网络告警语义</b> 与 <b>大语言模型（LLM）</b> 结合，
用于复杂通信网络的理解、预测与拓扑重建。
</p>

<p>
<b>网络拓扑</b> · <b>链路预测</b> · <b>节点恢复</b> · <b>告警语义</b> · <b>LLM 拓扑重建</b>
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

**网络结构 + 告警语义 → GNN 表征学习 → 网络预测 → LLM 推理 → 拓扑恢复**

<br/>

**简体中文** · [English](./README_EN.md)

</div>

---

## ⚡ 这是什么项目？

现代通信网络天然可以表示成图：

- **设备**是节点；
- **物理 / 逻辑连接**是边；
- **位置、角色等属性**描述结构；
- **告警与运行事件**提供语义上下文。

本项目研究一个核心问题：

> **当网络拓扑发生缺失、异常或故障时，图结构学习与大语言模型推理能否互相补充？**

**GNN_LLM_EXP** 实现了两条互补的智能化路径：

| 引擎 | 学习 / 推理对象 | 任务 |
|---|---|---|
| 🧠 **GNN 引擎** | 图结构 + 告警语义向量 | 链路预测、缺失节点识别 / 恢复 |
| ✨ **LLM 引擎** | 拓扑描述 + 运维告警 + 场景知识 | 拓扑恢复与语义推理 |

仓库覆盖完整实验链路：

**拓扑预处理 → 告警模拟 → 语义编码 → 图学习 → 推理 → 可视化 → LLM Prompt 生成 → 拓扑恢复评测**

---

# 🧬 系统架构

~~~mermaid
flowchart LR
    T["🌐 GraphML 网络拓扑<br/>节点 · 链路 · 地理属性"]
    A["🚨 网络告警<br/>CPU · 时延 · 接口 · 流量"]

    A --> AE["📝 告警聚合"]
    AE --> SE["🔤 Sentence Transformer<br/>all-MiniLM-L6-v2"]
    SE --> SEM["告警语义向量"]

    T --> SF["📐 结构特征<br/>Internal · Latitude · Longitude"]
    SF --> FUSE["⚡ 结构 + 语义融合"]
    SEM --> FUSE

    FUSE --> GNN["🧠 GNN 编码器<br/>GraphSAGE / GAT"]
    GNN --> LP["🔗 链路预测"]
    GNN --> NR["🧩 缺失节点恢复"]

    T --> SCENE["🎭 场景映射<br/>Wireless / Transport"]
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

一个网络节点不仅由“它连着谁”决定。

本项目将以下结构信息：

- Internal / External 标记
- Latitude
- Longitude

与网络运维告警的语义表示进行融合。

节点告警首先按设备聚合，再通过 **all-MiniLM-L6-v2** 编码为语义向量。

因此，每个节点同时拥有：

> **它在哪里、如何连接**

以及：

> **它正在发生什么**

两类信息。

</td>
<td width="50%" valign="top">

## ✨ LLM 拓扑推理

LLM 分支会把抽象网络拓扑转换为更贴近真实网络的设备场景。

**无线场景**

BBU · RRU · SW

**传输场景**

OLT · ONU · SW

实验会主动删除部分节点或链路，再向 LLM 提供：

- 当前残缺拓扑；
- 设备告警；
- Few-shot 示例；
- 严格的结构化输出要求。

目标是让模型推断并恢复缺失网络结构。

</td>
</tr>

<tr>
<td width="50%" valign="top">

## 🔗 链路预测

给定一个网络图，GNN 预测两个节点之间是否应该存在连接。

基线模型包含：

- GraphSAGE 编码
- Transformer 边交互
- Bilinear 双线性打分
- MLP 预测头

评测指标：

**ROC-AUC · Average Precision**

</td>
<td width="50%" valign="top">

## 🧩 缺失节点恢复

增强模型采用：

- Cross-Modal Fusion
- Multi-head Graph Attention
- GraphNorm
- Node Classification Head

部分节点被人工 Mask 后，模型尝试根据图结构与告警语义判断哪些节点发生缺失。

评测指标：

**Accuracy · Precision · Recall · F1**

</td>
</tr>
</table>

---

# 🧠 GNN 模型设计

## 基线模型：GraphSAGE + Transformer

模型文件：

~~~text
src/model.py
~~~

~~~mermaid
flowchart LR
    X["节点特征<br/>结构 + 告警语义"]
    X --> S1["GraphSAGE"]
    S1 --> S2["GraphSAGE"]
    S2 --> Z["节点表示"]

    Z --> C["节点对拼接"]
    C --> TR["Transformer Encoder"]
    Z --> BI["Bilinear Interaction"]

    TR --> M["融合"]
    BI --> M
    M --> P["链路存在概率"]

    Z --> N["节点分类头"]
    N --> R["缺失 / 正常"]
~~~

模型提供两个任务接口：

~~~text
forward_link()  → 正 / 负边得分
forward_node()  → 节点二分类 logits
~~~

---

## 增强模型：Cross-Modal Attention + GAT

模型文件：

~~~text
src/model_plus.py
~~~

增强模型显式拆分：

~~~text
结构特征 × 语义特征
~~~

先完成跨模态融合，再进行图传播。

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

    Z --> NODE["节点恢复 Head"]
    Z --> LINK["Bilinear 链路解码器"]
~~~

这个设计体现了项目最核心的思想：

> **拓扑结构告诉模型“节点之间是什么关系”，告警文本告诉模型“每个节点正在经历什么”。**

---

# 🚨 告警语义建模

网络告警由：

~~~text
src/simulate_alerts.py
~~~

生成。

当前模拟的主要事件包括：

| 告警类型 | 含义 |
|---|---|
| **interface_down** | 网络接口 / 端口异常 |
| **link_latency** | 链路往返时延异常 |
| **traffic_spike** | 带宽利用率突然升高 |
| **cpu_high** | CPU 负载过高 |
| **memory_full** | 内存压力过高 |

输出：

~~~text
alerts/node_alerts.csv
~~~

之后运行 **src/encode.py** 完成：

1. 按节点聚合多条告警；
2. 使用 SentenceTransformer 编码文本；
3. 保存节点级语义向量。

输出：

~~~text
processed/node_alerts_agg.csv
processed/node_alert_emb.pt
processed/node_ids.pt
~~~

---

# 🌐 网络拓扑数据

仓库中保留了一批 **GraphML** 格式的网络拓扑。

示例：

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

每个文件对应一个独立网络图实例。

数据构建流程会筛选连通图，并主要处理：

~~~text
5 ≤ 节点数量 ≤ 200
~~~

的网络，同时对结构特征进行标准化，再与节点告警语义向量融合。

---

# 🚀 快速开始

## 1. 克隆仓库

~~~bash
git clone https://github.com/Xu0225/GNN_LLM_EXP.git
cd GNN_LLM_EXP
~~~

## 2. 安装环境

~~~bash
pip install -r requirements.txt
~~~

环境快照包含：

~~~text
PyTorch
PyTorch Geometric
NetworkX
Sentence Transformers
Plotly
scikit-learn
LLM / Transformers 相关依赖
~~~

> 当前 requirements.txt 来自原始开发环境，其中包含 CUDA / GPU 相关 PyTorch 包。在不同 CUDA 版本或纯 CPU 环境下，建议先安装与本机匹配的 PyTorch。

---

# 🧪 GNN 实验流程

## Step 1：生成网络告警

~~~bash
python src/simulate_alerts.py
~~~

输出：

~~~text
alerts/node_alerts.csv
~~~

---

## Step 2：编码告警语义

~~~bash
python src/encode.py
~~~

内部使用：

~~~text
SentenceTransformer("all-MiniLM-L6-v2")
~~~

将自然语言告警转换成节点级稠密向量。

---

## Step 3：构建图数据

~~~bash
python src/create_dataset.py
~~~

数据流：

~~~text
GraphML
   ↓
拓扑筛选 / 结构特征标准化
   ↓
结构特征
   +
告警语义向量
   ↓
PyTorch Geometric Data
   ↓
节点 Mask / 恢复标签
~~~

处理结果保存在：

~~~text
processed/
~~~

仓库中同时保留了用于链路预测实验的 *_linkpred.pt 数据。

---

## Step 4：训练链路预测模型

~~~bash
python src/train_link_pred.py
~~~

训练目标：

~~~text
真实链路 → 1
负采样边 → 0
~~~

输出指标：

~~~text
ROC-AUC
Average Precision
~~~

模型权重：

~~~text
models/linkpred_best.pt
~~~

---

## Step 5：训练缺失节点恢复模型

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

模型权重：

~~~text
models/noderec_best.pt
~~~

当前仓库已保留两类训练完成的模型权重。

---

## Step 6：可视化节点恢复

~~~bash
python src/infer.py
~~~

当前推理脚本会区分：

- 实际缺失节点；
- 模型预测缺失节点；
- 其他正常节点。

并使用 Plotly 生成交互式网络图。

> 当前拓扑通过 src/infer.py 开头的 **topo** 变量指定。

---

# ✨ LLM 拓扑恢复

LLM 实验位于：

~~~text
llm_test/
~~~

与 GNN 分支不同，该部分有意保持 **模型供应商无关（Provider-Agnostic）**：仓库主要负责构造拓扑推理 Prompt 和评测模型输出，而不是绑定某一个商业 LLM API。

## 1. 将抽象拓扑转换为真实网络场景

~~~bash
python llm_test/real_data_proc/trans_to_realsenerio.py --input topologies/Abilene.graphml --scenario wireless --internal_only --drop_orig_label --seed 42 --output llm_test/real_data_proc/wireless_scenario.graphml
~~~

支持：

| 场景 | 设备类型 |
|---|---|
| **Wireless** | BBU · RRU · SW |
| **Transport** | OLT · ONU · SW |

---

## 2. 构造残缺拓扑与 LLM Prompt

~~~bash
python llm_test/real_data_proc/generate_prompt.py --gml llm_test/real_data_proc/wireless_scenario.graphml --remove_ratio 0.15 --mode node --scenario wireless --with_alerts --alert_mode aligned --out llm_test/real_data_proc/wireless_prompt.txt --mask_out llm_test/real_data_proc/wireless_mask.graphml --gt_json llm_test/real_data_proc/wireless_gt.json --seed 42
~~~

会生成三个关键实验文件：

~~~text
wireless_prompt.txt      → 提供给 LLM 的 Prompt
wireless_mask.graphml    → 被破坏后的残缺拓扑
wireless_gt.json         → 隐藏 Ground Truth
~~~

模型输出要求：

~~~json
{
  "removed_nodes": [],
  "removed_edges": [],
  "full_graphml": "...恢复后的 GraphML..."
}
~~~

---

## 3. 评估 LLM 恢复结果

将 LLM 输出保存到：

~~~text
llm_test/model_output.json
~~~

随后运行：

~~~bash
cd llm_test
python eval.py
~~~

评测器分别比较节点和边：

| 对象 | 指标 |
|---|---|
| **节点** | Precision · Recall · F1 |
| **边** | Precision · Recall · F1 |

同时会并排展示：

**原始拓扑 vs LLM 恢复拓扑**

用于直观检查恢复效果。

---

# 🔄 两种网络智能范式

~~~mermaid
flowchart TB
    P["残缺 / 异常网络"]
    P --> G["🧠 GNN 路径"]
    P --> L["✨ LLM 路径"]

    G --> G1["学习图的隐空间表示"]
    G1 --> G2["预测链路 / 缺失节点"]
    G2 --> G3["机器学习指标评测"]

    L --> L1["将拓扑转换为语言上下文"]
    L1 --> L2["结合拓扑 + 告警进行推理"]
    L2 --> L3["生成恢复后的 GraphML"]

    G3 --> C["🔬 对比与融合"]
    L3 --> C
~~~

这个项目真正值得继续研究的问题并不是简单的：

> **“GNN 和 LLM 谁更好？”**

而是：

> **“哪些信息应该从图结构中学习，哪些信息应该由语言语义推理，两者应该在哪一层融合？”**

---

# 📊 评测矩阵

| 任务 | 输入 | 模型 | 主要指标 |
|---|---|---|---|
| 🔗 链路预测 | 拓扑 + 语义特征 | GraphSAGE Fusion | ROC-AUC, AP |
| 🧩 节点恢复 | Mask 图 + 语义特征 | Cross-modal GAT | Accuracy, Precision, Recall, F1 |
| ✨ LLM 恢复 | 残缺拓扑 + 告警 | 外部 LLM | Node / Edge Precision, Recall, F1 |
| 📈 可视化 | 图 + 预测结果 | NetworkX / Plotly | 拓扑级定性分析 |

---

# 📂 仓库结构

~~~text
GNN_LLM_EXP/
│
├── src/
│   ├── simulate_alerts.py        # 网络告警模拟
│   ├── encode.py                 # 告警聚合 + MiniLM 编码
│   ├── create_dataset.py         # 构建 PyG 数据
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
├── topologies/                  # GraphML 拓扑数据
├── alerts/                      # 节点告警
├── processed/                   # PyG 数据与语义向量
├── models/
│   ├── linkpred_best.pt
│   └── noderec_best.pt
├── Aarnet_linkpred_top20.csv
├── requirements.txt
├── readme.md                    # 中文版 / GitHub 默认展示
└── README_EN.md                 # 英文版
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
| 指标评测 | **scikit-learn** |
| 可视化 | **Plotly · Matplotlib** |
| LLM 实验 | **Prompt Engineering · Structured JSON · GraphML Reconstruction** |

</div>

---

# 🔬 可继续扩展的研究方向

### 1. GNN → LLM

把 GNN 的高置信度预测结果作为结构化证据提供给 LLM，而不是让 LLM 完全依赖原始拓扑进行猜测。

### 2. LLM → GNN

利用 LLM 将真实告警、工单、维护日志、设备描述等非结构化信息转换成更丰富的图节点特征。

### 3. Graph + Language 联合推理

从当前松耦合的双路径实验，进一步升级为图语言模型、Graph-RAG 或联合图语义推理架构。

### 4. 面向真实网络运维

将模拟告警扩展为：

~~~text
Telemetry
Logs
SNMP
Performance Counters
Tickets
Fault Events
~~~

进一步验证真实网络故障下的拓扑恢复能力。

---

# ⚠️ 实验说明

本仓库属于研究原型，而不是已经产品化的软件库。

当前仍保留一些原始实验特征：

- 部分推理参数直接在 Python 文件中配置；
- 同时保留基线 GNN 与增强 GNN；
- 为方便复现实验，仓库中保存了部分 .pt 数据和训练权重；
- LLM 部分没有绑定具体模型供应商；
- requirements.txt 是完整开发环境快照，依赖范围大于项目最小运行依赖。

这些设计主要用于保留原实验过程，并方便继续研究和迭代。

---

<div align="center">

## 🌐 From Topology to Intelligence

**网络结构告诉我们“系统如何连接”。**  
**运行告警告诉我们“系统正在发生什么”。**  
**图学习与语言推理把二者连接起来。**

<br/>

⭐ 如果这个项目对你的研究有帮助，欢迎 Star。

<br/>

[English README](./README_EN.md)

<br/><br/>

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:7C3AED,50:2563EB,100:00C9A7&height=130&section=footer"/>

</div>
