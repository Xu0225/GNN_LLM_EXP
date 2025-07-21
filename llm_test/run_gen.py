#!/usr/bin/env python3
import os
import sys
import subprocess

# —— 顶部配置区 ——
TOPO_NAME     = "Aarnet"           # 拓扑名称（对应 <TOPO_NAME>.graphml）
GML_DIR       = "./topologies"       # 存放 GraphML 文件的目录
OUT_DIR       = "./prompts"          # 输出 Prompt 文本的目录
REMOVE_RATIO  = 0.02                 # 故障率（节点/边删除比例）
MODE          = "edge"               # 删除模式：node 或 edge
SCENARIO      = "wireless"           # 场景：wireless 或 transport

def main():
    # 1. 确保输出目录存在
    os.makedirs(OUT_DIR, exist_ok=True)

    # 2. 构造输入/输出路径
    script_dir   = os.path.dirname(os.path.abspath(__file__))
    gml_path     = os.path.join(GML_DIR, f"{TOPO_NAME}.graphml")
    output_path  = os.path.join(OUT_DIR, f"{TOPO_NAME}_prompt.txt")

    # 3. 组装命令
    cmd = [
        sys.executable,
        os.path.join(script_dir, "prompt_gen.py"),
        "--graphml_path",    gml_path,
        "--remove_ratio",    str(REMOVE_RATIO),
        "--mode",            MODE,
        "--scenario",        SCENARIO,
        "--output_path",     output_path
    ]

    # 4. 执行并打印日志
    print("运行命令:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=script_dir)
    if result.returncode != 0:
        print(f"调用 generate_prompt.py 出错，返回码: {result.returncode}")
        sys.exit(result.returncode)

    print("Prompt 生成完毕，文件路径：", output_path)

if __name__ == "__main__":
    main()
