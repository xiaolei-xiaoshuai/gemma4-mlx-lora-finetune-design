#!/usr/bin/env python3
"""抽样 + 格式归一：DocVQA / 代码 / 文档 -> messages，打 source 标签，存中间文件。"""
import json, random, os
random.seed(42)
BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, "raw")
OUT = os.path.join(BASE, "parts")
os.makedirs(OUT, exist_ok=True)

def read_jsonl(p, n=None):
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

# ---------- 1) DocVQA：从 data_full/train.jsonl (15000) 抽 10500 ----------
docvqa = read_jsonl(os.path.join(BASE, "..", "data_full", "train.jsonl"))
docvqa = docvqa[:15000]
docvqa_sample = random.sample(docvqa, 10500)
with open(os.path.join(OUT, "docvqa_10500.jsonl"), "w", encoding="utf-8") as f:
    for d in docvqa_sample:
        d["source"] = "docvqa"
        f.write(json.dumps(d, ensure_ascii=False) + "\n")
print("DocVQA ->", len(docvqa_sample))

# ---------- 2) 代码：CodeExercise 27224 -> 抽 2400，chat_rounds(human/bot) -> messages(user/assistant) ----------
code = read_jsonl(os.path.join(RAW, "CodeExercise-Python-27k.jsonl"))
code_sample = random.sample(code, 2400)
n_conv = 0
with open(os.path.join(OUT, "code_2400.jsonl"), "w", encoding="utf-8") as f:
    for d in code_sample:
        msgs = []
        for r in d.get("chat_rounds", []):
            role = "user" if r.get("role") == "human" else "assistant"
            msgs.append({"role": role, "content": r.get("content", "").strip()})
        # 只保留成对且非空的
        msgs = [m for m in msgs if m["content"]]
        if len(msgs) >= 2:
            n_conv += 1
        obj = {"messages": msgs, "source": "code"}
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
print("代码 ->", len(code_sample), "(有效成对:", n_conv, ")")

# ---------- 3) 文档：HUST 718 -> 抽 600，conversations -> messages ----------
hust = read_jsonl(os.path.join(RAW, "hust_handbook.jsonl"))
hust_sample = random.sample(hust, min(600, len(hust)))
n_doc = 0
with open(os.path.join(OUT, "docs_600.jsonl"), "w", encoding="utf-8") as f:
    for d in hust_sample:
        conv = d.get("conversations", [])
        msgs = [{"role": ("user" if c.get("role") in ("user","human") else "assistant"),
                 "content": c.get("content", "").strip()} for c in conv]
        msgs = [m for m in msgs if m["content"]]
        if len(msgs) >= 2:
            n_doc += 1
        obj = {"messages": msgs, "source": "docs"}
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
print("文档 ->", len(hust_sample), "(有效成对:", n_doc, ")")

# ---------- 预览各部分 1 条 ----------
for p in ["docvqa_10500","code_2400","docs_600"]:
    with open(os.path.join(OUT, p+".jsonl"), encoding="utf-8") as f:
        d = json.loads(f.readline())
    print(f"\n== {p} 示例 (轮数 {len(d['messages'])}) ==")
    for m in d["messages"][:2]:
        print(f"  [{m['role']}] {m['content'][:90]}")
