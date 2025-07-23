# GNN_LLM_EXP

本项目结合了图神经网络（GNN）与大语言模型（LLM），用于复杂网络的链路预测、节点推荐、异常检测等任务。项目涵盖了数据处理、模型训练、推理、可视化、告警分析等完整流程，支持多种真实网络拓扑数据，适用于网络科学、通信网络、复杂网络分析等领域的研究与应用。

---

## 目录结构

```markdown
GNN_LLM_EXP/
├── alerts/
├── llm_test/
│   ├── real_data_proc/
│   │   ├── net_topologies/
│   │   └── topologies/
│   ├── restored/
│   ├── prompts/
│   ├── eval_data/
│   └── topologies/
├── models/
├── processed/
├── src/
├── topologies/
```

---

## 主要功能
- **链路预测（Link Prediction）**  
  基于 GNN 对网络拓扑进行链路预测，辅助网络结构优化与演化分析。
- **节点推荐（Node Recommendation）**  
  针对不同网络，训练节点推荐模型，支持多种评估与可视化方式。
- **异常检测与告警分析**  
  结合告警数据，分析网络异常节点，辅助网络运维与安全。
- **大语言模型（LLM）辅助**  
  利用 LLM 进行 prompt 生成、自动化评估、结果解释等。
- **多网络拓扑支持**  
  支持数十种真实或仿真的网络拓扑（.graphml），可扩展性强。
- **可视化**  
  提供多种可视化脚本，直观展示预测结果与网络结构。
---

## 安装与环境

1. **克隆项目**
   ```bash
   git clone https://github.com/Xu0225/GNN_LLM_EXP.git
   cd GNN_LLM_EXP
   ```

2. **安装依赖**
   推荐使用 Python 3.8+，并建议使用虚拟环境。
   ```bash
   pip install -r requirements.txt
   ```
---

## 快速开始

### 1. 数据准备

- 网络拓扑数据位于 `topologies/`，格式为 `.graphml`。
- 告警数据位于 `alerts/node_alerts.csv`。
- 处理后模型权重在 `processed/`，训练好的模型在 `models/`。

### 2. 训练模型

以链路预测为例：

```bash
python src/train_link_pred.py --config configs/link_pred.yaml
```

节点推荐：

```bash
python src/train_node_rec.py --config configs/node_rec.yaml
```

### 3. 推理与评估

```bash
python src/infer.py --model models/linkpred_best.pt --data topologies/xxx.graphml
python src/evaluate_link_pred.py --result results/xxx_pred.csv
```

### 4. 可视化

```bash
python src/visualize_linkpred.py --input results/xxx_pred.csv
```

### 5. LLM 提示词生成

```bash
# 原始拓扑转换
python trans_to_realsenerio.py
  --input original_topology.graphml
  --scenario wireless
  --internal_only
  --drop_orig_label
  --output wireless_real.graphml
```
```bash
# 提示词生成
python generate_prompt.py
  --gml ./net_topologies/wireless_scenario.graphml 
  --remove_ratio 0.15
  --mode node
  --scenario wireless
  --with_alerts
  --alert_mode aligned
  --out ./prompts/Chinanet_prompt.txt
  --mask_out ./topologies/Chinanet_mask.graphml
  --gt_json ./topologies/Chinanet_gt.json
  --seed 42
```

---

## 主要脚本说明

- `src/train_link_pred.py`：链路预测模型训练
- `src/train_node_rec.py`：节点推荐模型训练
- `src/infer.py`：模型推理
- `src/visualize_linkpred.py`、`src/visualize_noderec.py`：结果可视化
- `src/model.py`、`src/model_plus.py`：模型结构定义
- `src/create_dataset.py`：数据集构建
- `src/simulate_alerts.py`：告警模拟
- `llm_test/generate_prompt.py`、`llm_test/prompt_gen.py`：LLM prompt 生成
- `llm_test/eval.py`：LLM 结果评估

---

## 数据说明

- **topologies/**：包含多种网络拓扑的 `.graphml` 文件
- **alerts/node_alerts.csv**：节点级别的告警/异常数据
- **processed/**、**models/**：不同网络/任务的模型权重

---
