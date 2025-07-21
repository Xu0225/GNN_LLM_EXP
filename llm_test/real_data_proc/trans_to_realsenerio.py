#!/usr/bin/env python3
"""
trans_to_realsenerio.py  — 场景化转换 + 过滤

功能  ⮕  read ➜ rename ➜ filter ➜ write
---------------------------------------
1. 读取原始 .graphml 拓扑文件。
2. 将节点重命名为符合场景 (wireless / transport) 的设备名前缀。
3. **可选**：
   ▸ 仅保留 Internal==1 的节点（过滤国际/测试节点）。
   ▸ 去掉 orig_label 等中间字段，使结果图只含真实场景字段。
4. 可为节点添加角色字段 role（radio_unit / baseband_unit / switch）。
5. 输出新的场景化拓扑 .graphml。

用法示例
---------
python trans_to_realsenerio.py \
    --input original.graphml \
    --scenario wireless \
    --internal_only \
    --drop_orig_label \
    --output wireless_scenario.graphml
"""
import argparse
import random
from pathlib import Path
import networkx as nx

PREFIX_MAP = {
    "wireless":  ["BBU", "RRU", "SW"],
    "transport": ["OLT", "ONU", "SW"],
}

INTERNAL_KEY = "Internal"  # GraphML 中记录内部/外部的字段 (d28)
ORIG_KEY     = "orig_label"  # 若需删除原标签，可通过 --drop_orig_label


def filter_internal_nodes(G: nx.Graph) -> None:
    """删除 Internal==0 的节点及相关边。"""
    to_remove = [n for n, data in G.nodes(data=True)
                 if str(data.get(INTERNAL_KEY, "1")) == "0"]
    G.remove_nodes_from(to_remove)


def remap_and_annotate(G: nx.Graph, scenario: str, drop_orig_label: bool):
    prefixes = PREFIX_MAP[scenario]
    mapping = {}
    for idx, node_id in enumerate(list(G.nodes()), 1):
        prefix = random.choice(prefixes)
        new_name = f"{prefix}-{idx:03d}"
        mapping[node_id] = new_name

        # 添加/更新角色字段
        role = {
            "RRU": "radio_unit",
            "BBU": "baseband_unit",
            "SW":  "switch",
            "OLT": "olt",
            "ONU": "onu",
        }[prefix]
        G.nodes[node_id]["role"] = role

        # 记录原始 label 便于溯源
        if not drop_orig_label:
            G.nodes[node_id][ORIG_KEY] = node_id

    nx.relabel_nodes(G, mapping, copy=False)

    # 可选：真正删除 orig_label 字段
    if drop_orig_label:
        for _, data in G.nodes(data=True):
            data.pop(ORIG_KEY, None)


def transform_graph(input_path: Path, scenario: str, internal_only: bool,
                    drop_orig_label: bool, output_path: Path, seed: int | None):
    if seed is not None:
        random.seed(seed)

    G = nx.read_graphml(input_path)

    if internal_only:
        filter_internal_nodes(G)

    remap_and_annotate(G, scenario, drop_orig_label)

    nx.write_graphml(G, output_path)
    print(f"✅ Scene topology saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert original GraphML into a realistic wireless/transport scenario with optional filtering.")
    parser.add_argument("--input", required=True, help="Path to original .graphml")
    parser.add_argument("--scenario", choices=list(PREFIX_MAP.keys()), default="wireless")
    parser.add_argument("--internal_only", action="store_true",
                        help="Keep only nodes where Internal==1 (remove external/test links)")
    parser.add_argument("--drop_orig_label", action="store_true",
                        help="Remove orig_label field from final output")
    parser.add_argument("--output", required=True, help="Path to output .graphml")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    transform_graph(Path(args.input), args.scenario, args.internal_only,
                    args.drop_orig_label, Path(args.output), args.seed)

if __name__ == "__main__":
    main()