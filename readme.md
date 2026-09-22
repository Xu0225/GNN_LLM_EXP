<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:00C9A7,45:2563EB,100:7C3AED&height=230&section=header&text=GNN%20%C3%97%20LLM%20for%20Network%20Intelligence&fontSize=42&fontColor=FFFFFF&fontAlignY=35&desc=Structure-aware%20Graph%20Learning%20%2B%20Semantic%20Reasoning%20for%20Topology%20Recovery&descAlignY=55&descSize=16&animation=fadeIn"/>

# 🌐 GNN × LLM EXP

### Graph Intelligence for Network Topology Prediction, Recovery & Reasoning

<p>
A research prototype that combines <b>Graph Neural Networks</b>, <b>alert semantics</b>, and <b>Large Language Models</b>
to understand, predict, and reconstruct complex communication networks.
</p>

<p>
<b>Topology Graphs</b> · <b>Link Prediction</b> · <b>Node Recovery</b> · <b>Alarm Semantics</b> · <b>LLM Topology Reconstruction</b>
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

**Graph Structure + Alarm Semantics → GNN Representation Learning → Network Prediction → LLM Reasoning → Topology Recovery**

</div>

---

## ⚡ What is this project?

Modern communication networks are naturally represented as graphs:

- **devices** are nodes,
- **physical / logical connections** are edges,
- **location and role attributes** describe structure,
- **alarms and operational events** provide semantic context.

This project explores a simple but powerful idea:

> **Can structural graph learning and language-model reasoning complement each other when a network topology becomes incomplete or abnormal?**

**GNN_LLM_EXP** implements two complementary intelligence paths:

| Engine | What it learns / reasons about | Tasks |
|---|---|---|
| 🧠 **GNN Engine** | graph structure + semantic alert embeddings | link prediction, missing-node detection / recovery |
| ✨ **LLM Engine** | topology description + operational alerts + scenario knowledge | topology reconstruction and reasoning |

The repository includes the full experimental chain: **topology preprocessing → alert simulation → semantic encoding → graph learning → inference → visualization → LLM prompt generation → topology recovery evaluation**.

---

# 🧬 System Architecture

~~~mermaid
flowchart LR
    T["🌐 GraphML Topologies<br/>nodes · links · geo attributes"]
    A["🚨 Network Alerts<br/>CPU · latency · interface · traffic"]

    A --> AE["📝 Alert Aggregation"]
    AE --> SE["🔤 Sentence Transformer<br/>all-MiniLM-L6-v2"]
    SE --> SEM["Semantic Embeddings"]

    T --> SF["📐 Structural Features<br/>Internal · Latitude · Longitude"]
    SF --> FUSE["⚡ Structure + Semantics"]
    SEM --> FUSE

    FUSE --> GNN["🧠 GNN Encoder<br/>GraphSAGE / GAT"]
    GNN --> LP["🔗 Link Prediction"]
    GNN --> NR["🧩 Missing Node Recovery"]

    T --> SCENE["🎭 Scenario Mapping<br/>Wireless / Transport"]
    SCENE --> MASK["💥 Topology Masking<br/>Remove Nodes / Edges"]
    MASK --> PROMPT["📜 Prompt Builder<br/>Topology + Alerts + Few-shot"]
    PROMPT --> LLM["✨ Large Language Model"]
    LLM --> RESTORE["🔧 Restored GraphML"]

    LP --> VIZ["📊 Evaluation & Visualization"]
    NR --> VIZ
    RESTORE --> EV["✅ Node / Edge Recovery Metrics"]

    style GNN fill:#2563EB,color:#fff
    style LLM fill:#7C3AED,color:#fff
    style FUSE fill:#00A98F,color:#fff
~~~

---

# 🔥 Core Ideas

<table>
<tr>
<td width="50%" valign="top">

## 🧠 Structure-Semantic Fusion

A node is not represented only by graph connectivity.

The project combines structural information such as:

- Internal / external indicator
- Latitude
- Longitude

with semantic representations derived from operational alarms.

Alarm texts are aggregated per node and encoded with:

`all-MiniLM-L6-v2`

This produces a graph where every node contains both:

> **where it is / how it connects**

and

> **what is happening to it**

</td>
<td width="50%" valign="top">

## ✨ LLM Topology Reasoning

The LLM branch converts abstract topology graphs into realistic network scenarios such as:

**Wireless**

`BBU · RRU · SW`

**Transport**

`OLT · ONU · SW`

Nodes or links are intentionally removed and the LLM receives:

- the remaining topology,
- recent alarms,
- few-shot examples,
- a strict structured output schema.

Its goal is to reconstruct the missing network structure.

</td>
</tr>

<tr>
<td width="50%" valign="top">

## 🔗 Link Prediction

Given an incomplete graph, the GNN estimates whether a connection should exist between two nodes.

The baseline model uses:

- GraphSAGE encoding
- Transformer-based edge interaction
- Bilinear edge scoring
- MLP prediction head

Evaluation:

`ROC-AUC` · `Average Precision`

</td>
<td width="50%" valign="top">

## 🧩 Missing Node Recovery

The enhanced model uses:

- Cross-modal fusion
- Multi-head Graph Attention
- GraphNorm
- Node classification head

A subset of nodes is masked from the graph and the model learns to identify likely missing nodes from topology and semantic context.

Evaluation:

`Accuracy` · `Precision` · `Recall` · `F1`

</td>
</tr>
</table>

---

# 🧠 GNN Model Design

## Baseline — GraphSAGE + Transformer

`src/model.py`

~~~mermaid
flowchart LR
    X["Node Features<br/>Structure + Semantics"]
    X --> S1["GraphSAGE"]
    S1 --> S2["GraphSAGE"]
    S2 --> Z["Node Embeddings"]

    Z --> C["Pair Concatenation"]
    C --> TR["Transformer Encoder"]
    Z --> BI["Bilinear Interaction"]

    TR --> M["Fusion"]
    BI --> M
    M --> P["Link Probability"]

    Z --> N["Node Classification Head"]
    N --> R["Missing / Normal"]
~~~

The model exposes two task heads:

~~~text
forward_link()  → positive / negative edge scores
forward_node()  → node-level binary logits
~~~

---

## Enhanced — Cross-Modal Attention + GAT

`src/model_plus.py`

The enhanced node-recovery model explicitly separates:

~~~text
structural features  ×  semantic features
~~~

and performs cross-modal fusion before graph propagation.

~~~mermaid
flowchart LR
    ST["📐 Structural Vector"]
    SM["💬 Alert Semantic Vector"]

    ST --> Q["Query Projection"]
    SM --> KV["Key / Value Projection"]

    Q --> ATT["Cross-Modal Attention"]
    KV --> ATT

    ATT --> CAT["Feature Fusion"]
    SM --> CAT

    CAT --> G1["Multi-head GAT"]
    G1 --> GN1["GraphNorm + ELU"]
    GN1 --> G2["GAT"]
    G2 --> GN2["GraphNorm + ELU"]
    GN2 --> Z["Network-aware Embedding"]

    Z --> NODE["Node Recovery Head"]
    Z --> LINK["Bilinear Link Decoder"]
~~~

This architecture is the most distinctive part of the repository:

> **network topology tells the model how nodes relate; alarm language tells it what each node is experiencing.**

---

# 🚨 Alarm Intelligence

Network alarms are generated in `src/simulate_alerts.py`.

Supported simulated events include:

| Alert Type | Example Meaning |
|---|---|
| `interface_down` | device interface or port failure |
| `link_latency` | abnormal round-trip delay |
| `traffic_spike` | sudden bandwidth utilization increase |
| `cpu_high` | processing overload |
| `memory_full` | memory pressure |

The generated alarm stream is stored as:

~~~text
alerts/node_alerts.csv
~~~

Then `src/encode.py`:

1. aggregates all alarms belonging to the same node,
2. encodes the alarm text using **SentenceTransformer**,
3. saves semantic tensors into the processed dataset.

Outputs:

~~~text
processed/node_alerts_agg.csv
processed/node_alert_emb.pt
processed/node_ids.pt
~~~

---

# 🌐 Topology Corpus

The repository contains a collection of real-world-style communication network topologies in **GraphML** format.

Examples include:

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

Each topology becomes an independent graph instance.

The dataset builder keeps connected graphs with approximately:

~~~text
5 ≤ number of nodes ≤ 200
~~~

and standardizes structural attributes before merging semantic features.

---

# 🚀 Quick Start

## 1. Clone

~~~bash
git clone https://github.com/Xu0225/GNN_LLM_EXP.git
cd GNN_LLM_EXP
~~~

## 2. Environment

~~~bash
pip install -r requirements.txt
~~~

The environment snapshot includes PyTorch, PyTorch Geometric, NetworkX, Sentence Transformers, Plotly, scikit-learn and the LLM ecosystem used during experimentation.

> The committed `requirements.txt` contains GPU-specific PyTorch packages from the original development environment.  
> On another CUDA / CPU platform, install a matching PyTorch build first if necessary.

---

# 🧪 GNN Pipeline

## Step 1 — Simulate Network Alerts

~~~bash
python src/simulate_alerts.py
~~~

Output:

~~~text
alerts/node_alerts.csv
~~~

---

## Step 2 — Encode Alert Semantics

~~~bash
python src/encode.py
~~~

This uses:

~~~text
SentenceTransformer("all-MiniLM-L6-v2")
~~~

to transform alarm language into dense node-level semantic vectors.

---

## Step 3 — Build Graph Data

~~~bash
python src/create_dataset.py
~~~

The dataset construction pipeline:

~~~text
GraphML
   ↓
filter / normalize
   ↓
structural node attributes
   +
alert semantic embeddings
   ↓
PyTorch Geometric Data
   ↓
node masking / recovery labels
~~~

Processed topology tensors are stored in:

~~~text
processed/
~~~

The repository also contains prepared `*_linkpred.pt` artifacts for link-prediction experiments.

---

## Step 4 — Train Link Prediction

~~~bash
python src/train_link_pred.py
~~~

Training objective:

~~~text
positive links  → 1
negative links  → 0
~~~

Reported metrics:

~~~text
ROC-AUC
Average Precision
~~~

Model output:

~~~text
models/linkpred_best.pt
~~~

---

## Step 5 — Train Missing-Node Recovery

~~~bash
python src/train_node_rec.py
~~~

Reported validation metrics:

~~~text
Accuracy
Precision
Recall
F1
~~~

Model output:

~~~text
models/noderec_best.pt
~~~

Both trained model checkpoints are already included in this repository.

---

## Step 6 — Visualize Recovery

~~~bash
python src/infer.py
~~~

The inference script compares:

- actual missing nodes,
- predicted missing nodes,
- unaffected network nodes,

and renders the result as an interactive Plotly graph.

> The current script selects the topology through the `topo` variable near the beginning of `src/infer.py`.

---

# ✨ LLM Topology Recovery

The LLM side is an independent experimental pipeline under:

~~~text
llm_test/
~~~

Unlike the GNN branch, it is deliberately **provider-agnostic**: the repository focuses on prompt construction and output evaluation rather than binding the experiment to one specific commercial LLM API.

## 1. Convert an Abstract Graph into a Network Scenario

~~~bash
python llm_test/real_data_proc/trans_to_realsenerio.py   --input topologies/Abilene.graphml   --scenario wireless   --internal_only   --drop_orig_label   --seed 42   --output llm_test/real_data_proc/wireless_scenario.graphml
~~~

Supported scenarios:

| Scenario | Device Vocabulary |
|---|---|
| **Wireless** | BBU · RRU · SW |
| **Transport** | OLT · ONU · SW |

---

## 2. Create a Damaged Topology + Reasoning Prompt

~~~bash
python llm_test/real_data_proc/generate_prompt.py   --gml llm_test/real_data_proc/wireless_scenario.graphml   --remove_ratio 0.15   --mode node   --scenario wireless   --with_alerts   --alert_mode aligned   --out llm_test/real_data_proc/wireless_prompt.txt   --mask_out llm_test/real_data_proc/wireless_mask.graphml   --gt_json llm_test/real_data_proc/wireless_gt.json   --seed 42
~~~

This produces three important experimental artifacts:

~~~text
wireless_prompt.txt      → prompt given to the LLM
wireless_mask.graphml    → damaged / incomplete topology
wireless_gt.json         → hidden ground truth
~~~

The prompt asks the model to return:

~~~json
{
  "removed_nodes": [],
  "removed_edges": [],
  "full_graphml": "... reconstructed GraphML ..."
}
~~~

---

## 3. Evaluate the Restored Topology

Place the LLM response in:

~~~text
llm_test/model_output.json
~~~

and run:

~~~bash
cd llm_test
python eval.py
~~~

The evaluator compares the reconstructed graph against the original topology at both levels:

| Level | Metrics |
|---|---|
| **Nodes** | Precision · Recall · F1 |
| **Edges** | Precision · Recall · F1 |

It also renders the **original topology** and **restored topology** side by side.

---

# 🔄 Two Intelligence Paradigms

~~~mermaid
flowchart TB
    P["Incomplete / Abnormal Network"]

    P --> G["🧠 GNN Path"]
    P --> L["✨ LLM Path"]

    G --> G1["Learn latent graph representation"]
    G1 --> G2["Predict links / missing nodes"]
    G2 --> G3["Statistical ML metrics"]

    L --> L1["Translate topology into language"]
    L1 --> L2["Reason from graph + alarms"]
    L2 --> L3["Generate reconstructed GraphML"]

    G3 --> C["🔬 Compare Approaches"]
    L3 --> C
~~~

The research question is not simply **“GNN or LLM?”**

A more interesting direction is:

> **What should be learned from graph structure, what should be inferred from operational semantics, and where should the two meet?**

---

# 📊 Evaluation Matrix

| Task | Input | Model | Primary Metrics |
|---|---|---|---|
| 🔗 Link Prediction | topology + semantic features | GraphSAGE fusion model | ROC-AUC, AP |
| 🧩 Node Recovery | masked graph + semantic features | Cross-modal GAT | Accuracy, Precision, Recall, F1 |
| ✨ LLM Recovery | incomplete topology + alarms | external LLM | Node / Edge Precision, Recall, F1 |
| 📈 Visualization | graph + prediction | NetworkX / Plotly | qualitative topology inspection |

---

# 📂 Repository Structure

~~~text
GNN_LLM_EXP/
│
├── src/
│   ├── simulate_alerts.py        # Generate operational alarms
│   ├── encode.py                 # Alarm aggregation + MiniLM embeddings
│   ├── create_dataset.py         # Build PyG graph datasets
│   ├── model.py                  # GraphSAGE + Transformer fusion model
│   ├── model_plus.py             # Cross-modal attention + GAT model
│   ├── train_link_pred.py        # Link prediction training
│   ├── train_node_rec.py         # Missing-node recovery training
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
├── topologies/                  # GraphML network topology corpus
├── alerts/                      # Node-level alarm data
├── processed/                   # Prepared PyG tensors + embeddings
├── models/
│   ├── linkpred_best.pt
│   └── noderec_best.pt
│
├── Aarnet_linkpred_top20.csv
├── requirements.txt
└── readme.md
~~~

---

# 🛠️ Tech Stack

<div align="center">

| Layer | Technology |
|---|---|
| Graph Learning | **PyTorch · PyTorch Geometric** |
| GNN Operators | **GraphSAGE · GAT · GraphNorm** |
| Semantic Encoding | **Sentence Transformers · MiniLM** |
| Graph Processing | **NetworkX · GraphML** |
| Classical Metrics | **scikit-learn** |
| Visualization | **Plotly · Matplotlib** |
| LLM Experiment | **Prompt Engineering · Structured JSON · GraphML Reconstruction** |

</div>

---

# 🔬 Research Directions

This prototype naturally opens several extensions:

### 1. GNN → LLM Context

Use high-confidence GNN predictions as structured evidence for the LLM instead of asking the LLM to reconstruct the network from raw topology alone.

### 2. LLM → GNN Semantics

Use an LLM to transform raw alarms, tickets, maintenance logs, and device descriptions into richer node features.

### 3. Joint Graph-Language Reasoning

Replace the loose two-stage connection with graph-aware language models or graph retrieval pipelines.

### 4. Real Network Operations

Extend simulated alarms to:

~~~text
telemetry · logs · SNMP · performance counters · tickets · fault events
~~~

and evaluate topology recovery under real operational failures.

---

# ⚠️ Experimental Notes

This repository is a research prototype rather than a packaged production library.

A few scripts reflect the original experimental workflow:

- some inference settings are selected directly inside Python files,
- both baseline and enhanced GNN implementations are retained,
- prepared `.pt` datasets and model checkpoints are committed for reproducibility,
- the LLM experiment does not hard-code a model provider or API call,
- `requirements.txt` is a full environment snapshot and is broader than the minimum runtime dependency set.

These choices preserve the original experiments while keeping the project easy to inspect and extend.

---

<div align="center">

## 🌐 From topology to intelligence

**Structure tells us how a network is connected.**  
**Alarms tell us what the network is experiencing.**  
**Graph learning and language reasoning connect the two.**

<br/>

⭐ If this project is useful for your research, consider starring the repository.

<br/>

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:7C3AED,50:2563EB,100:00C9A7&height=130&section=footer"/>

</div>
