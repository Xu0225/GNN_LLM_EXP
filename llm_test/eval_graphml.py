#!/usr/bin/env python3
# eval_graphml.py

import argparse
import xml.etree.ElementTree as ET
import re
from datetime import datetime

def parse_graphml(path):
    try:
        ns = {'g': 'http://graphml.graphdrawing.org/xmlns'}
        tree = ET.parse(path)
        root = tree.getroot()
        id2lab = {}
        for nd in root.findall('.//g:node', ns):
            nid = nd.get('id')
            lab = nd.get('label')
            if lab is None:
                data = nd.find('g:data[@key="label"]', ns)
                lab = data.text if data is not None else nid
            id2lab[nid] = lab
        nodes = set(id2lab.values())
        edges = set()
        for ed in root.findall('.//g:edge', ns):
            u, v = ed.get('source'), ed.get('target')
            edges.add(frozenset({id2lab.get(u, u), id2lab.get(v, v)}))
        return nodes, edges
    except ET.ParseError:
        text = open(path, 'r', encoding='utf-8', errors='ignore').read()
        id2lab = {}
        for m in re.finditer(r'<node\s+id="([^"]+)".*?<data\s+key="label">([^<]+)</data>', text, re.S):
            id2lab[m.group(1)] = m.group(2)
        nodes = set(id2lab.values())
        edges = set()
        for m in re.finditer(r'<edge\s+source="([^"]+)"\s+target="([^"]+)"', text):
            u, v = m.group(1), m.group(2)
            edges.add(frozenset({id2lab.get(u, u), id2lab.get(v, v)}))
        return nodes, edges

def compute_metrics(orig, rec):
    tp = len(orig & rec)
    fp = len(rec - orig)
    fn = len(orig - rec)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec_ = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec_ / (prec + rec_) if (prec + rec_) else 0.0
    acc = tp / len(orig) if orig else 0.0
    return tp, fp, fn, prec, rec_, f1, acc

def main():
    parser = argparse.ArgumentParser(description="Evaluate GraphML recovery")
    parser.add_argument('--original', required=True, help="Original GraphML file")
    parser.add_argument('--recovered', required=True, help="Recovered GraphML file")
    args = parser.parse_args()

    print(f"Evaluation time: {datetime.now().isoformat()}\n")
    orig_nodes, orig_edges = parse_graphml(args.original)
    rec_nodes, rec_edges   = parse_graphml(args.recovered)

    for title, oset, rset in [
        ("Node recovery", orig_nodes, rec_nodes),
        ("Edge recovery", orig_edges, rec_edges)
    ]:
        tp, fp, fn, prec, rec_, f1, acc = compute_metrics(oset, rset)
        print(f"{title}:")
        print(f"  True Positive : {tp}")
        print(f"  False Positive: {fp}")
        print(f"  False Negative: {fn}")
        print(f"  Precision     : {prec:.4f}")
        print(f"  Recall        : {rec_:.4f}")
        print(f"  F1 Score      : {f1:.4f}")
        print(f"  Accuracy      : {acc:.4f}\n")

if __name__ == '__main__':
    main()
