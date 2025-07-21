#!/usr/bin/env python3
import os
import subprocess

topo_name = "Chinanet"
# ========== 配置区（可根据需求修改） ==========
GML_PATH        = f"./topologies/{topo_name}.graphml"
REMOVE_RATIO    = 0.1
MODE            = "edge"
SCENARIO        = "transport"
WITH_ALERTS     = True
# ALERTS_CSV_PATH = "./alerts/node_alerts.csv"   # 若不加载真实告警，设为 None
ALERTS_CSV_PATH = None
OUTPUT_PATH     = f"./prompts/{topo_name}_prompt.txt"
GENERATE_SCRIPT = "generate_prompt.py"
# ========== End 配置区 ==========

# 确保输出目录存在
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# 构造命令
cmd = [
    "python", GENERATE_SCRIPT,
    "--gml", GML_PATH,
    "--remove_ratio", str(REMOVE_RATIO),
    "--mode", MODE,
    "--scenario", SCENARIO,
]
if WITH_ALERTS:
    cmd.append("--with_alerts")
    if ALERTS_CSV_PATH:
        cmd.extend(["--alerts_csv", ALERTS_CSV_PATH])
cmd.extend(["--out", OUTPUT_PATH])

# 执行
print("Running command:", " ".join(cmd))
subprocess.run(cmd, check=True)
print(f"Prompt 已生成：{OUTPUT_PATH}")
