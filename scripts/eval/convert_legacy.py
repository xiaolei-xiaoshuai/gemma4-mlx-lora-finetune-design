#!/usr/bin/env python3
"""把旧的扁平结果（{题id: 回答}，如 test_v1.json/test_v2.json/test_base.json）
转成 run_suite.py 的新格式（{meta, groups}），以便和 compare_tables.py 混用。
用法:
  python convert_legacy.py <old.json> <out.json> [模型标签]
旧格式里固定的 5 题映射到组：
  docvqa->docvqa, code->code, dlg1/dlg2/dlg3->dialogue
"""
import json, sys, os

old = sys.argv[1]
out = sys.argv[2]
label = sys.argv[3] if len(sys.argv) > 3 else os.path.splitext(os.path.basename(old))[0]

d = json.load(open(old, encoding="utf-8"))
# 旧 5 题 -> 组
mapping = {"docvqa": "docvqa", "code": "code", "dlg1": "dialogue", "dlg2": "dialogue", "dlg3": "dialogue"}

groups = {}
for key, resp in d.items():
    g = mapping.get(key, key)
    m = {
        "thought_leak": ("<|channel>thought" in resp) or ("|channel|thought" in resp) or ("<channel>" in resp),
        "code_fenced": "```" in resp,
        "n_chars": len(resp),
        "n_tokens_approx": len(resp) // 2,
    }
    groups.setdefault(g, {"description": {"docvqa": "文档问答", "code": "编程", "dialogue": "日常对话"}.get(g, g),
                          "questions": []})["questions"].append(
        {"id": key, "prompt": "(旧 5 题集)", "response": resp, "metrics": m})

for g in groups:
    qs = groups[g]["questions"]
    groups[g]["summary"] = {"n": len(qs),
                            "thought_leak": sum(1 for q in qs if q["metrics"]["thought_leak"]),
                            "code_fenced": sum(1 for q in qs if q["metrics"]["code_fenced"])}

outd = {"meta": {"model": label, "note": "converted from legacy flat json", "converted_from": old},
        "groups": groups}
json.dump(outd, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"已转换 {old} -> {out}（模型标签 {label}，组: {list(groups.keys())}）")
