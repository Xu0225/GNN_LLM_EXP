#!/usr/bin/env python3
"""
convert_json_to_graphml.py

将大模型输出（非严格 JSON 格式）文件中的 full_graphml 部分提取并保存为合法的 GraphML 文件。

使用说明：
1. 确保脚本与你的模型输出文件（如 model_output.json）在同一目录。
2. 修改 INPUT_FILE 为实际文件名（默认 model_output.json）。
3. 运行：
       python convert_json_to_graphml.py
4. 生成的 GraphML 文件为 restored_graph.graphml。
"""

import os
import re

# —— 配置区 ——
INPUT_FILE     = "model_output.json"       # 大模型输出文件名
OUTPUT_FILE    = "./restored/restored_graph.graphml"  # 输出的 GraphML 文件名

def main():
    # 检查文件存在性
    if not os.path.isfile(INPUT_FILE):
        print(f"未找到输入文件：{INPUT_FILE}")
        return

    # 读取整个文件文本
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        text = f.read()

    # 提取 <?xml ...?> 到 </graphml> 区间
    m = re.search(r'(<\?xml[\s\S]*?</graphml>)', text)
    if not m:
        print("未匹配到 GraphML 内容，请检查输入文件格式")
        return

    graphml_content = m.group(1)

    # 写入 .graphml 文件
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(graphml_content)

    print(f"已生成 GraphML 文件：{OUTPUT_FILE}")

if __name__ == "__main__":
    main()

