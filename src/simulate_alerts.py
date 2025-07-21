import os
import csv
import random
import networkx as nx

# 脚本文件位于 …/data/src，真实的 topologies 在 …/data/topologies
BASE_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW_DIR   = os.path.join(BASE_DIR, 'topologies')
ALERT_DIR = os.path.join(BASE_DIR, 'alerts')

os.makedirs(ALERT_DIR, exist_ok=True)


# 告警模板
TEMPLATES = {
    'interface_down': [
        "接口 {if_name} 在端口 {port} 出现断开",
        "设备接口{if_name}端口{port}状态 DOWN"
    ],
    'link_latency': [
        "与邻居节点 {nbr} 的链路延迟超过 {latency}ms",
        "检测到到节点{nbr}的往返延迟 RTT ~{latency}ms"
    ],
    'traffic_spike': [
        "入流量瞬时增长至 {rate}% 带宽利用率",
        "流量负载突增，当前占用 {rate}%"
    ],
    'cpu_high': [
        "CPU 利用率超过 {rate}%，可能存在过载风险",
        "处理器占用率 {rate}%，需关注性能瓶颈"
    ],
    'memory_full': [
        "内存使用率达 {rate}%，系统将触发垃圾回收",
        "RAM 占用 {rate}%，剩余不足"
    ]
}

def pick_interface(G, node):
    # 假定每个节点接口列表
    return f"eth{random.randint(0,3)}"

def simulate_one_alert(G, node, degree, centrality, nbrs):
    weights = {
        'interface_down':    1.0,
        'link_latency':      1.0,
        'traffic_spike':     centrality * 5,
        'cpu_high':          centrality * 3,
        'memory_full':       centrality * 2
    }
    if not nbrs:
        weights['link_latency'] = 0.0

    total = sum(weights.values())
    pick = random.uniform(0, total)
    cum = 0
    for typ, w in weights.items():
        cum += w
        if pick <= cum:
            template = random.choice(TEMPLATES[typ])
            if typ == 'interface_down':
                return template.format(
                    if_name=pick_interface(G, node),
                    port=random.randint(1, 48)
                )
            if typ == 'link_latency':
                if nbrs:
                    return template.format(
                        nbr=random.choice(nbrs),
                        latency=random.randint(100, 500)
                    )
                # 回退逻辑
                template = random.choice(TEMPLATES['interface_down'])
                return template.format(
                    if_name=pick_interface(G, node),
                    port=random.randint(1, 48)
                )
            # traffic_spike / cpu_high / memory_full
            return template.format(rate=random.randint(70, 100))
    return ""


def main():
    # 输出 CSV： node_id, alert_text
    with open(os.path.join(ALERT_DIR, 'node_alerts.csv'), 'w', newline='', encoding='utf-8') as fout:
        writer = csv.writer(fout)
        writer.writerow(['node_id', 'alert_text'])
        # 遍历所有图
        for fname in os.listdir(RAW_DIR):
            if not fname.endswith('.graphml'): continue
            G = nx.read_graphml(os.path.join(RAW_DIR, fname))
            # 计算中心性
            centrality = nx.degree_centrality(G)
            for node in G.nodes():
                deg = G.degree[node]
                nbrs = list(G.neighbors(node))
                # 每个节点模拟 1~3 条告警
                for _ in range(random.randint(1,3)):
                    alert = simulate_one_alert(G, node, deg, centrality[node], nbrs)
                    writer.writerow([node, alert])
    print("告警模拟完成，文件保存在 data/alerts/node_alerts.csv")

if __name__ == '__main__':
    main()
