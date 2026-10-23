#!/usr/bin/env python
"""守护清理器：训练时每 60s 扫一次 adapters_full，只保留最新 2 个编号 checkpoint。
避免删到正在写的文件（只删存在超过 5 分钟的旧文件）。"""
import glob, os, time, sys

DIR = "/Users/Admin/Desktop/微调和强化学习/gemma4-lora-pilot/adapters_full"
KEEP = 2
MIN_AGE_S = 300  # 只删 5 分钟前的

def run_once():
    files = sorted(glob.glob(os.path.join(DIR, "0*_adapters.safetensors")))
    if len(files) <= KEEP:
        return
    now = time.time()
    # 从旧的开始删，保留最新 KEEP 个
    to_delete = files[:-KEEP]
    freed = 0
    for f in to_delete:
        if now - os.path.getmtime(f) > MIN_AGE_S:
            sz = os.path.getsize(f)
            os.remove(f)
            freed += sz
    if freed:
        print(f"清理 {len(to_delete)} 个旧 checkpoint，释放 {freed/1024**3:.1f}G", flush=True)

while True:
    try:
        run_once()
    except Exception as e:
        pass
    time.sleep(60)
