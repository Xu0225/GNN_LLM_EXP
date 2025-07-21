#!/usr/bin/env python3
"""
generate_prompt.py（改进版）

支持生成与 Ground Truth 对齐的告警信息：
- 对被删除的节点优先生成告警（--alert_mode aligned）
- 或混合少量未删除节点（--alert_mode mixed）
- 默认行为与之前一致（--alert_mode random）
"""
import argparse
import random
import networkx as nx
import pandas as pd
from datetime import datetime, timedelta
import json

SCENARIO_PREFIXES = {
    'wireless': ['BBU', 'RRU', 'SW'],
    'transport': ['OLT', 'ONU', 'SW']
}

ALERT_TEMPLATES = [
    "CPU 利用率连续 {m} 分钟超过 {p}%，存在过载风险",
    "接口 {iface} 在端口 {port} 出现短时断开",
    "设备温度超过 {t}°C，可能导致性能不稳定"
]

def load_graph(path: str) -> nx.Graph:
    return nx.read_graphml(path)

def simulate_missing(graph: nx.Graph, remove_ratio: float, mode: str):
    G = graph.copy()
    if mode == 'edge':
        items = list(G.edges())
    else:
        items = list(G.nodes())
    k = max(1, int(len(items) * remove_ratio))
    removed = random.sample(items, k)
    if mode == 'edge':
        G.remove_edges_from(removed)
    else:
        G.remove_nodes_from(removed)
    return G, removed

def simulate_alerts_by_mode(G: nx.Graph, removed, mode: str, alert_mode: str, max_per_node=2):
    alerts = {}
    nodes = list(G.nodes())
    removed_nodes = removed if mode == 'node' else []
    alert_nodes = []
    if alert_mode == 'aligned':
        alert_nodes = removed_nodes
    elif alert_mode == 'mixed':
        noise = random.sample([n for n in nodes if n not in removed_nodes], max(1, len(removed_nodes) // 2))
        alert_nodes = removed_nodes + noise
    else:  # random
        k = max(1, int(len(nodes) * 0.15))
        alert_nodes = random.sample(nodes, k)
    for nid in alert_nodes:
        msgs = []
        for _ in range(random.randint(1, max_per_node)):
            tmpl = random.choice(ALERT_TEMPLATES)
            msgs.append(tmpl.format(
                m=random.randint(1,10),
                p=random.randint(80,98),
                peer=random.choice(nodes),
                lat=random.randint(100,600),
                iface=f"eth{random.randint(0,3)}",
                port=random.randint(1,48),
                t=random.randint(60,90)
            ))
        alerts[nid] = '；'.join(msgs)
    return alerts

def map_node_labels(graph: nx.Graph, scenario: str):
    prefixes = SCENARIO_PREFIXES.get(scenario, ['NODE'])
    mapping = {}
    for idx, nid in enumerate(graph.nodes(), 1):
        prefix = random.choice(prefixes)
        mapping[nid] = f"{prefix}-{idx:03d}"
    nx.relabel_nodes(graph, mapping, copy=False)

def build_prompt(graph: nx.Graph, alerts: dict = None):
    system = (
        "System:\n"
        "你是一名经验丰富的无线网络运维工程师，"
        "擅长根据设备告警信息和拓扑结构恢复网络拓扑。\n"
        "请严格按照示例的输出格式，仅返回 JSON 对象，无任何额外说明。"
    )
    examples = [
        {
            "graph": {
                "nodes": ["A", "B", "C", "D"],
                "edges": [["A", "B"], ["B", "C"], ["C", "D"]]
            },
            "alerts": {
                "C": "CPU 利用率连续 5 分钟超过 90%，存在过载风险"
            },
            "output": {
                "removed_nodes": [],
                "removed_edges": [["C", "D"]],
                "full_graphml": "<?xml version=\"1.0\"?>...完整 GraphML 文本..."
            }
        },
        {
            "graph": {
                "nodes": ["X", "Y", "Z"],
                "edges": [["X", "Y"], ["Y", "Z"], ["Z", "X"]]
            },
            "alerts": {},
            "output": {
                "removed_nodes": ["Z"],
                "removed_edges": [],
                "full_graphml": "<?xml version=\"1.0\"?>...完整 GraphML 文本..."
            }
        }
    ]

    ex_lines = ["\nExamples:"]
    for ex in examples:
        ex_lines.append("User:")
        ex_lines.append(f"Graph nodes: {ex['graph']['nodes']}")
        ex_lines.append(f"Graph edges: {ex['graph']['edges']}")
        ex_lines.append(f"Alerts: {ex['alerts']}")
        ex_lines.append("Expected JSON output:")
        ex_lines.append(f"{ex['output']}\n")
    user = ["User:", "现有一张无线网络的残缺拓扑图（因告警导致部分节点或链路失效）。", "节点及边如下："]
    for n in graph.nodes():
        user.append(f"- {n}: {list(graph.neighbors(n))}")
    if alerts:
        user.append("告警信息（仅保留“严重”级别＋最近 1 小时内摘要）：")
        now = datetime.now()
        for idx, (nid, txt) in enumerate(alerts.items(), 1):
            ts = (now - timedelta(minutes=random.randint(1, 59))).strftime("%Y-%m-%d %H:%M")
            user.append(f"{idx}. {ts}  {nid}: {txt}")
    user.append("请基于上述信息，输出 JSON：")
    user.append("{")
    user.append('  "removed_nodes": [...],')
    user.append('  "removed_edges": [[u,v],...],')
    user.append('  "full_graphml": "...完整 GraphML 文本..."')
    user.append("}")
    return "\n".join([system] + ex_lines + user)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gml", required=True)
    parser.add_argument("--remove_ratio", type=float, default=0.1)
    parser.add_argument("--mode", choices=["edge", "node"], required=True)
    parser.add_argument("--scenario", choices=list(SCENARIO_PREFIXES), default="wireless")
    parser.add_argument("--with_alerts", action="store_true")
    parser.add_argument("--alerts_csv", default=None)
    parser.add_argument("--alert_mode", choices=["random", "aligned", "mixed"], default="random")
    parser.add_argument("--out", default="prompt.txt")
    parser.add_argument("--mask_out", default=None)
    parser.add_argument("--gt_json", default=None)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    G_full = load_graph(args.gml)
    G_mask, removed = simulate_missing(G_full, args.remove_ratio, args.mode)
    map_node_labels(G_mask, args.scenario)

    if args.with_alerts:
        if args.alerts_csv:
            alerts = load_alerts_from_csv(args.alerts_csv)
        else:
            alerts = simulate_alerts_by_mode(G_mask, removed, args.mode, args.alert_mode)
    else:
        alerts = None

    prompt = build_prompt(G_mask, alerts)
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write(prompt)

    if args.mask_out:
        nx.write_graphml(G_mask, args.mask_out)
    if args.gt_json:
        gt = {"mode": args.mode, f"removed_{args.mode}s": removed}
        with open(args.gt_json, 'w', encoding='utf-8') as f:
            json.dump(gt, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
