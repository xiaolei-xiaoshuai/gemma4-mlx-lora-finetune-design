#!/usr/bin/env python3
"""组装 v2 混合数据集（50/25/25）:
  docvqa 7500 (从 parts/docvqa_10500 抽) + code 3750 (从 raw 27k 抽) + dialogue 3750
  -> data_mixed_v2/{train,valid,test}.jsonl，按 source 分层 90/5/5 切分
"""
import json, random, os, collections
random.seed(2025)
BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, "raw")
PARTS = os.path.join(BASE, "parts")
ROOT = os.path.dirname(BASE)
OUT = os.path.join(ROOT, "data_mixed_v2")
os.makedirs(OUT, exist_ok=True)

def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

# 1) docvqa: 10500 -> 7500
docvqa = read_jsonl(os.path.join(PARTS, "docvqa_10500.jsonl"))
docvqa_sample = random.sample(docvqa, 7500)
docvqa_sample = [{"messages": d["messages"], "source": "docvqa"} for d in docvqa_sample]
print("docvqa:", len(docvqa_sample))

# 2) code: 27224 -> 3750（chat_rounds human/bot -> messages user/assistant）
code = read_jsonl(os.path.join(RAW, "CodeExercise-Python-27k.jsonl"))
code_sample = random.sample(code, 3750)
n_conv = 0
code_rows = []
for d in code_sample:
    msgs = []
    for r in d.get("chat_rounds", []):
        role = "user" if r.get("role") == "human" else "assistant"
        msgs.append({"role": role, "content": (r.get("content") or "").strip()})
    msgs = [m for m in msgs if m["content"]]
    if len(msgs) >= 2:
        n_conv += 1
    code_rows.append({"messages": msgs, "source": "code"})
print("code:", len(code_rows), "(有效成对:", n_conv, ")")

# 3) dialogue: 3750
dialogue = read_jsonl(os.path.join(BASE, "dialogue_3750.jsonl"))
for d in dialogue:
    d["source"] = "dialogue"
print("dialogue:", len(dialogue))

# 分层 90/5/5
buckets = {"docvqa": docvqa_sample, "code": code_rows, "dialogue": dialogue}
SPLIT = {"train": [], "valid": [], "test": []}
for src, items in buckets.items():
    random.shuffle(items)
    n = len(items)
    nv, nt = round(n * 0.05), round(n * 0.05)
    train, valid, test = items[:n - nv - nt], items[n - nv - nt:n - nt], items[n - nt:]
    for it in train:  it["source"] = src
    for it in valid:  it["source"] = src
    for it in test:   it["source"] = src
    SPLIT["train"] += train; SPLIT["valid"] += valid; SPLIT["test"] += test
    print(f"{src}: train {len(train)} valid {len(valid)} test {len(test)}")

random.shuffle(SPLIT["train"]); random.shuffle(SPLIT["valid"]); random.shuffle(SPLIT["test"])
for name, rows in SPLIT.items():
    p = os.path.join(OUT, f"{name}.jsonl")
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"写出 {name}: {len(rows)}")

# 配比报告
for name in ["train", "valid", "test"]:
    c = collections.Counter(r["source"] for r in SPLIT[name])
    tot = sum(c.values())
    print(f"{name:6s} 配比: " + "  ".join(f"{k} {100*v/tot:.1f}%" for k, v in c.items()))
