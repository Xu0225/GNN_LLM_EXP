#!/usr/bin/env python3
"""
generate_prompt.py

一键生成符合指定模板的拓扑还原 Prompt 文本，并包含 Few-Shot 示例和明确的输出格式：
1. 加载完整 GML
2. 随机删除生成残缺图或节点
3. 可选加载真实告警或模拟告警文本
4. 场景化映射节点标签
5. 构建包含 System、Examples 和 User 段的 Prompt
"""
import argparse
import random
import networkx as nx
import pandas as pd
from datetime import datetime, timedelta

# 可配置的场景类型和对应节点前缀
SCENARIO_PREFIXES = {
    'wireless': ['BBU', 'RRU', 'SW'],
    'transport': ['OLT', 'ONU', 'SW']
}

# 模拟告警模板
ALERT_TEMPLATES = [
    "CPU 利用率连续 {m} 分钟超过 {p}%，存在过载风险",
    "与邻居节点 {peer} 的链路延迟连续超过 {lat} ms",
    "接口 {iface} 在端口 {port} 出现短时断开",
    "设备温度超过 {t}°C，可能导致性能不稳定"
]

def load_graph(path: str) -> nx.Graph:
    return nx.read_graphml(path)

def simulate_missing(graph: nx.Graph, remove_ratio: float, mode: str):
    G = graph.copy()
    if mode == 'edge':
        items = list(G.edges())
        k = max(1, int(len(items) * remove_ratio))
        removed = random.sample(items, k)
        G.remove_edges_from(removed)
    else:
        items = list(G.nodes())
        k = max(1, int(len(items) * remove_ratio))
        removed = random.sample(items, k)
        G.remove_nodes_from(removed)
    return G, removed

def load_alerts_from_csv(path: str) -> dict:
    df = pd.read_csv(path, dtype=str)
    return df.groupby('node_id')['alert'].agg(lambda x: '；'.join(x)).to_dict()

def simulate_alerts(graph: nx.Graph, max_per_node: int = 3) -> dict:
    alerts = {}
    nodes = list(graph.nodes())
    for nid in nodes:
        cnt = random.randint(0, max_per_node)
        if cnt > 0:
            msgs = []
            for _ in range(cnt):
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

def map_node_labels(graph: nx.Graph, scenario: str) -> None:
    prefixes = SCENARIO_PREFIXES.get(scenario, ['NODE'])
    mapping = {}
    for idx, nid in enumerate(graph.nodes(), 1):
        prefix = random.choice(prefixes)
        mapping[nid] = f"{prefix}-{idx:03d}"
    nx.relabel_nodes(graph, mapping, copy=False)

def build_prompt(graph: nx.Graph, alerts: dict = None) -> str:
    # 1. System 段
    system = (
        "System:\n"
        "你是一名经验丰富的无线网络运维工程师，"
        "擅长根据设备告警信息和拓扑结构恢复网络拓扑。\n"
        "请严格按照示例的输出格式，仅返回 JSON 对象，无任何额外说明。\n"
    )
    # 2. Few-Shot 示例
    examples = [
        {
            "graph": {
                "nodes": ["A", "B", "C", "D"],
                "edges": [["A","B"],["B","C"],["C","D"]]
            },
            "alerts": {
                "C": "CPU 利用率连续 5 分钟超过 90%，存在过载风险"
            },
            "removed": {
                "nodes": [],
                "edges": [["C","D"]]
            },
            "output": {
                "removed_nodes": [],
                "removed_edges": [["C","D"]],
                "full_graphml": "<?xml version=\"1.0\"?>...完整 GraphML 文本..."
            }
        },
        {
            "graph": {
                "nodes": ["X", "Y", "Z"],
                "edges": [["X","Y"],["Y","Z"],["Z","X"]]
            },
            "alerts": {},
            "removed": {
                "nodes": ["Z"],
                "edges": []
            },
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
        if ex['alerts']:
            ex_lines.append(f"Alerts: {ex['alerts']}")
        else:
            ex_lines.append("Alerts: {}")
        ex_lines.append("Expected JSON output:")
        ex_lines.append(f"{ex['output']}\n")
    # 3. User 段（实际任务）
    user = ["User:"]
    user.append("现有一张无线网络的残缺拓扑图（因告警导致部分节点或链路失效）。")
    user.append("节点及边如下：")
    for n in graph.nodes():
        user.append(f"- {n}: {list(graph.neighbors(n))}")
    if alerts:
        user.append("告警信息（仅保留“严重”级别＋最近 1 小时内摘要）：")
        now = datetime.now()
        idx = 1
        for nid, txt in alerts.items():
            minutes_ago = random.randint(1, 59)
            ts = (now - timedelta(minutes=minutes_ago)).strftime("%Y-%m-%d %H:%M")
            user.append(f"{idx}. {ts}  {nid}: {txt}")
            idx += 1
    user.append("请基于上述信息，输出 JSON：")
    user.append("{")
    user.append('  "removed_nodes": [...],')
    user.append('  "removed_edges": [[u,v],...],')
    user.append('  "full_graphml": "...完整 GraphML 文本..."')
    user.append("}")
    return "\n".join([system] + ex_lines + user)

def main():
    parser = argparse.ArgumentParser(description="生成包含 Few-Shot 和输出格式的 Prompt")
    parser.add_argument("--gml",    required=True, help="完整 GML 文件路径")
    parser.add_argument("--remove_ratio", type=float, default=0.1, help="删除比例")
    parser.add_argument("--mode",   choices=["edge","node"], required=True, help="删除模式")
    parser.add_argument("--scenario", choices=list(SCENARIO_PREFIXES.keys()), default='wireless', help="场景类型")
    parser.add_argument("--with_alerts", action="store_true", help="加载真实或模拟告警")
    parser.add_argument("--alerts_csv", default=None, help="真实告警 CSV 路径")
    parser.add_argument("--out",     default="prompt.txt", help="输出 Prompt 文件")
    args = parser.parse_args()

    G_full = load_graph(args.gml)
    G_mask, _ = simulate_missing(G_full, args.remove_ratio, args.mode)
    map_node_labels(G_mask, args.scenario)

    if args.with_alerts and args.alerts_csv:
        alerts = load_alerts_from_csv(args.alerts_csv)
    elif args.with_alerts:
        alerts = simulate_alerts(G_mask)
    else:
        alerts = None

    prompt = build_prompt(G_mask, alerts)
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write(prompt)
    print(f"已生成 Prompt: {args.out}")

if __name__ == "__main__":
    main()
