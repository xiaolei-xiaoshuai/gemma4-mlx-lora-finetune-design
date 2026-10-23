#!/usr/bin/env python3
"""组装混合数据集：docvqa10500 + code2400 + docs600 + dialogue1500 -> data_mixed/
按 source 分层切 90/5/5，保证每个 split 都保持 70/20/10 配比。
输出目录: gemma4-lora-pilot/data_mixed/{train,valid,test}.jsonl  (MLX 直接读这个目录)"""
import json, random, os, collections
random.seed(42)
BASE = os.path.dirname(os.path.abspath(__file__))
PARTS = os.path.join(BASE, "parts")
ROOT = os.path.dirname(BASE)                 # gemma4-lora-pilot
OUT = os.path.join(ROOT, "data_mixed")
os.makedirs(OUT, exist_ok=True)

def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

# 载入四块（dialogue 已在 dialogue_1500.jsonl）
buckets = {
    "docvqa":  read_jsonl(os.path.join(PARTS, "docvqa_10500.jsonl")),
    "code":    read_jsonl(os.path.join(PARTS, "code_2400.jsonl")),
    "docs":    read_jsonl(os.path.join(PARTS, "docs_600.jsonl")),
    "dialogue":read_jsonl(os.path.join(BASE, "dialogue_1500.jsonl")),
}
for k, v in buckets.items():
    print(f"载入 {k}: {len(v)}")

# 分层切分：每个 source 90/5/5
SPLIT = {"train": [], "valid": [], "test": []}
stats = collections.Counter()
for src, items in buckets.items():
    random.shuffle(items)
    n = len(items)
    nv, nt = round(n * 0.05), round(n * 0.05)
    train, valid, test = items[:n - nv - nt], items[n - nv - nt:n - nt], items[n - nt:]
    for it in train:  it["source"] = src
    for it in valid:  it["source"] = src
    for it in test:   it["source"] = src
    SPLIT["train"] += train; SPLIT["valid"] += valid; SPLIT["test"] += test
    stats[f"{src}_train"] = len(train); stats[f"{src}_valid"] = len(valid); stats[f"{src}_test"] = len(test)

# 写文件（MLX 布局）+ 整体再 shuffle train
random.shuffle(SPLIT["train"]); random.shuffle(SPLIT["valid"]); random.shuffle(SPLIT["test"])
for name, rows in SPLIT.items():
    p = os.path.join(OUT, f"{name}.jsonl")
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"写出 {name}: {len(rows)}")

# 校验 + 配比统计
def mix_report(rows):
    c = collections.Counter(r["source"] for r in rows)
    tot = sum(c.values())
    return {k: f"{v}({100*v//tot}%)" for k, v in c.items()}
print("\n===== 各 split 配比 =====")
for name in ["train", "valid", "test"]:
    print(f"{name:6s} {len(SPLIT[name]):6d}  配比: " + "  ".join(mix_report(SPLIT[name]).values()))
print("\n===== train 首行示例 =====")
print(json.dumps(SPLIT["train"][0], ensure_ascii=False)[:200])
