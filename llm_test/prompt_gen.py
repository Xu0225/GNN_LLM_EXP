#!/usr/bin/env python3
"""
generate_prompt.py

一键生成符合指定模板的拓扑还原 Prompt 文本，并包含 Few-Shot 示例和明确的输出格式：
1. 加载完整 GraphML
2. 基于模拟告警驱动节点/边删除（贴近真实场景）
3. 场景化映射节点标签
4. 构建包含 System、Examples 和 User 段的 Prompt
"""
import argparse
import random
import networkx as nx
import os

# —— 默认配置 ——
DEFAULT_REMOVE_RATIO = 0.02    # 故障率 2%
DEFAULT_MODE = 'node'          # 删除模式：node 或 edge
DEFAULT_SCENARIO = 'wireless'  # 场景：wireless 或 transport
MAX_ALERTS = 5                 # 单次最多生成 5 条告警

def load_graph(path):
    """从 GraphML 文件加载网络拓扑。"""
    return nx.read_graphml(path)

def simulate_alerts_by_structure(graph: nx.Graph, scenario: str):
    """
    根据节点度数和场景，从高风险节点池中随机挑选少量节点生成告警。
    wireless: 丢包率、过载风险
    transport: 链路延迟、接口断开
    """
    degrees = dict(graph.degree())
    avg_degree = sum(degrees.values()) / len(degrees) if degrees else 0

    # 高风险节点：度数 > 1.5 × 平均度
    high_risk = [n for n, d in degrees.items() if d > avg_degree * 1.5]
    # 若过少，则拓展到度数 > 平均度
    if len(high_risk) < MAX_ALERTS:
        high_risk = [n for n, d in degrees.items() if d > avg_degree]
    # 若依然为空，则所有节点都可能告警
    if not high_risk:
        high_risk = list(graph.nodes())

    # 最多选 MAX_ALERTS 个节点
    alert_nodes = random.sample(high_risk, min(len(high_risk), MAX_ALERTS))

    alerts = {}
    for nid in alert_nodes:
        if scenario == 'wireless':
            alerts[nid] = random.choice([
                f"节点{nid}检测到丢包率超过5%，影响无线链路质量",
                f"节点{nid}的连接数{degrees[nid]}高于平均度{avg_degree:.1f}，存在过载风险"
            ])
        else:  # transport
            alerts[nid] = random.choice([
                f"节点{nid}检测到链路延迟连续超过100ms，存在通信风险",
                f"节点{nid}接口短时断开，影响传输稳定性"
            ])
    return alerts

def simulate_missing_by_alert(graph: nx.Graph, alerts: dict, remove_ratio: float):
    """
    仅从告警节点中随机删除少部分，模拟真实场景中告警不一定全失效。
    """
    alert_nodes = list(alerts.keys())
    k = max(1, int(len(alert_nodes) * remove_ratio))
    removed = random.sample(alert_nodes, k)
    Gm = graph.copy()
    Gm.remove_nodes_from(removed)
    return Gm, removed

def simulate_edge_missing_by_alert(graph: nx.Graph, alerts: dict, remove_ratio: float):
    """
    仅删除与告警节点关联的部分链路，若无关联链路则退回随机删除。
    """
    alert_nodes = set(alerts.keys())
    candidate_edges = [e for e in graph.edges() if e[0] in alert_nodes or e[1] in alert_nodes]
    if not candidate_edges:
        # 回退到随机边删除
        items = list(graph.edges())
        k = max(1, int(len(items) * remove_ratio))
        removed = random.sample(items, k)
    else:
        k = max(1, int(len(candidate_edges) * remove_ratio))
        removed = random.sample(candidate_edges, k)
    Gm = graph.copy()
    Gm.remove_edges_from(removed)
    return Gm, removed

def map_node_labels(graph: nx.Graph, scenario: str):
    """
    场景化映射节点标签。wireless 场景下随机使用 SW, RRU, BBU 前缀；
    transport 场景下随机使用 SW, OLT, RT 前缀。
    """
    prefixes = ['SW','RRU','BBU'] if scenario == 'wireless' else ['SW','OLT','RT']
    mapping = {}
    for idx, node in enumerate(graph.nodes()):
        prefix = random.choice(prefixes)
        mapping[node] = f"{prefix}-{idx:03d}"
    new_graph = nx.relabel_nodes(graph, mapping, copy=True)
    return new_graph, mapping

def build_prompt(graph: nx.Graph, alerts: dict):
    """
    构建完整 Prompt 文本，包括 System, Examples, User 三部分。
    """
    # —— Few-Shot 示例 ——
    ex1_nodes = ['A', 'B', 'C']
    ex1_edges = [['A','B'], ['B','C']]
    ex1_alerts = {'B': 'CPU利用率超过90%'}
    ex1_output = "{'removed_nodes': [], 'removed_edges': [['B','C']], 'full_graphml': '...完整 GraphML 文本...'}"

    ex2_nodes = ['X', 'Y']
    ex2_edges = [['X','Y']]
    ex2_alerts = {}
    ex2_output = "{'removed_nodes': ['Y'], 'removed_edges': [], 'full_graphml': '...完整 GraphML 文本...'}"

    # —— 系统说明 ——
    system_txt = (
        "System:\n"
        "你是一名经验丰富的无线网络运维工程师，擅长根据设备告警信息和拓扑结构恢复网络拓扑。\n"
        "请严格按照示例的输出格式，仅返回 JSON 对象，无任何额外说明。\n\n"
    )

    prompt = system_txt
    prompt += "Example 1:\n"
    prompt += f"Graph nodes: {ex1_nodes}\n"
    prompt += f"Graph edges: {ex1_edges}\n"
    prompt += f"Alerts: {ex1_alerts}\n"
    prompt += f"Expected JSON output: {ex1_output}\n\n"

    prompt += "Example 2:\n"
    prompt += f"Graph nodes: {ex2_nodes}\n"
    prompt += f"Graph edges: {ex2_edges}\n"
    prompt += f"Alerts: {ex2_alerts}\n"
    prompt += f"Expected JSON output: {ex2_output}\n\n"

    # —— 用户输入 ——
    nodes_list = list(graph.nodes())
    edges_list = [list(e) for e in graph.edges()]
    prompt += "User:\n"
    prompt += f"Graph nodes: {nodes_list}\n"
    prompt += f"Graph edges: {edges_list}\n"
    prompt += f"Alerts: {alerts}\n"
    prompt += "Expected JSON output:"
    return prompt

def main():
    parser = argparse.ArgumentParser(
        description="生成符合模板的拓扑还原 Prompt 文本"
    )
    parser.add_argument(
        "--graphml_path", "-g", required=True,
        help="输入完整拓扑 GraphML 文件路径"
    )
    parser.add_argument(
        "--remove_ratio", "-r", type=float, default=DEFAULT_REMOVE_RATIO,
        help="故障删除比例，默认 0.02"
    )
    parser.add_argument(
        "--mode", "-m", choices=['node','edge'], default=DEFAULT_MODE,
        help="删除模式：node 或 edge"
    )
    parser.add_argument(
        "--scenario", "-s", choices=['wireless','transport'], default=DEFAULT_SCENARIO,
        help="场景：wireless 或 transport"
    )
    parser.add_argument(
        "--output_path", "-o", required=True,
        help="输出 Prompt 文本文件路径"
    )
    args = parser.parse_args()

    # 1. 加载完整图
    G = load_graph(args.graphml_path)

    # 2. 生成模拟告警（贴近真实场景）
    alerts_orig = simulate_alerts_by_structure(G, args.scenario)

    # 3. 基于告警驱动的节点/边删除
    if args.mode == 'node':
        G_mask, removed_orig = simulate_missing_by_alert(G, alerts_orig, args.remove_ratio)
    else:
        G_mask, removed_orig = simulate_edge_missing_by_alert(G, alerts_orig, args.remove_ratio)

    # 4. 场景化节点标签映射
    G_labeled, mapping = map_node_labels(G_mask, args.scenario)

    # 5. 将原始告警和删除列表映射到新标签
    alerts_new = { mapping[n]: alerts_orig[n]
                   for n in alerts_orig if n in mapping }
    if args.mode == 'node':
        removed_new = [mapping[n] for n in removed_orig if n in mapping]
    else:
        removed_new = [[mapping[u], mapping[v]]
                       for (u, v) in removed_orig
                       if u in mapping and v in mapping]

    # 6. 构建 Prompt 并写入文件
    prompt = build_prompt(G_labeled, alerts_new)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w', encoding='utf-8') as f:
        f.write(prompt)

    print(f"已生成 Prompt: {args.output_path}")

if __name__ == "__main__":
    main()
