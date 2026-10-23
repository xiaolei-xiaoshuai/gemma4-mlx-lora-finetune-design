#!/usr/bin/env python3
"""多领域测试运行器。
用法:
  python run_suite.py <model_path> <out_json> [suite.json] [group ...]
  - 不指定 group = 跑全部组；指定 = 只跑这些组
  输出 out_json: {meta, groups:{g:{desc, questions:[{id,prompt,response,metrics}]}}}
"""
import json, sys, time
import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

MODEL = sys.argv[1]
OUT = sys.argv[2]
SUITE = sys.argv[3] if len(sys.argv) > 3 else "test_suite.json"
ONLY = sys.argv[4:] if len(sys.argv) > 4 else []

suite = json.load(open(SUITE, encoding="utf-8"))
groups = suite["groups"]
if ONLY:
    groups = {g: groups[g] for g in ONLY if g in groups}

S = suite.get("meta", {}).get("sampling", {})
temp, top_p, max_tok = S.get("temp", 0.7), S.get("top_p", 0.95), S.get("max_tokens", 512)

print(f"加载模型 {MODEL} ...", flush=True)
model, tokenizer = load(MODEL)
print("模型加载完成", flush=True)
sampler = make_sampler(temp=temp, top_p=top_p)


def metrics(resp: str):
    low = resp
    leak = ("channel>thought" in low) or ("|channel|thought" in low) or ("<channel>" in low)
    return {
        "thought_leak": leak,
        "code_fenced": "```" in resp,
        "n_chars": len(resp),
        "n_tokens_approx": len(resp) // 2,
    }


out = {"meta": {"model": MODEL, "suite": SUITE, "groups": list(groups.keys()),
                "sampling": {"temp": temp, "top_p": top_p, "max_tokens": max_tok}},
       "groups": {}}

for gname, g in groups.items():
    gdesc = g.get("description", "")
    qout = []
    print(f"\n##### 组 [{gname}] {gdesc}（{len(g['questions'])} 题）", flush=True)
    for q in g["questions"]:
        prompt_src = q["prompt"]
        if isinstance(prompt_src, str):
            msgs = [{"role": "user", "content": prompt_src}]
        else:
            msgs = prompt_src
        prompt = tokenizer.apply_chat_template(msgs, add_generation_prompt=True)
        t0 = time.time()
        resp = generate(model, tokenizer, prompt, max_tokens=max_tok, sampler=sampler, verbose=False)
        dt = time.time() - t0
        m = metrics(resp)
        print(f"  [{q['id']}] {dt:.1f}s leak={m['thought_leak']} fence={m['code_fenced']} {m['n_chars']}字", flush=True)
        qout.append({"id": q["id"], "prompt": prompt_src, "response": resp,
                     "metrics": m, "seconds": round(dt, 1)})
        mx.metal.clear_cache()
    # 组级汇总
    n = len(qout)
    leak = sum(1 for x in qout if x["metrics"]["thought_leak"])
    fence = sum(1 for x in qout if x["metrics"]["code_fenced"])
    out["groups"][gname] = {"description": gdesc, "questions": qout,
                            "summary": {"n": n, "thought_leak": leak, "code_fenced": fence}}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f"\n已保存 {OUT}")
