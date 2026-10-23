"""DocVQA 领域能力评测：EM / F1 / BLEU / ROUGE-L
- fused（全量 LoRA）：360 条测试集全跑
- base（底座）：抽 30 条对照
参考：DocVQA 官方评测的 ANLS 之外，标准 SQuAD 式 EM/F1 是通用做法
"""
import json, re, time
import mlx.core as mx
from mlx_lm import load, generate as lm_generate

BASE = "/Users/Admin/.lmstudio/models/lmstudio-community/gemma-4-26B-A4B-it-QAT-MLX-4bit"
FUSED = "/Users/Admin/.lmstudio/models/local/gemma-4-26B-A4B-docvqa-lora-full"
TEST = "/Users/Admin/Desktop/微调和强化学习/gemma4-lora-pilot/data_full/test.jsonl"

PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)

def normalize(s: str) -> str:
    s = s.lower()
    s = PUNCT_RE.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()

def token_f1(pred: str, gold: str):
    pt, gt = normalize(pred).split(), normalize(gold).split()
    if not pt or not gt:
        return 0.0
    overlap = len(set(pt) & set(gt))
    return 2.0 * overlap / (len(pt) + len(gt)) if overlap else 0.0

samples = [json.loads(l) for l in open(TEST) if l.strip()]
print(f"测试集共 {len(samples)} 条", flush=True)

def run_model(name, path, idxs, max_tokens):
    print(f"=== 加载 {name} ...", flush=True)
    model, tokenizer = load(path)
    mx.eval(model)
    preds = []
    t0 = time.time()
    for i in idxs:
        s = samples[i]
        msgs = s["messages"]
        pid = tokenizer.apply_chat_template(msgs[:-1], add_generation_prompt=True, tokenize=True)
        out = lm_generate(model, tokenizer, prompt=pid, max_tokens=max_tokens, verbose=False)
        preds.append(out.strip())
        if (i + 1) % 40 == 0:
            print(f"    {name} 已生成 {i+1}/{max(idxs)+1} ({time.time()-t0:.0f}s)", flush=True)
    print(f"=== {name} 生成完成 {time.time()-t0:.0f}s", flush=True)
    del model, tokenizer
    mx.clear_cache()

    golds = [normalize(samples[i]["messages"][-1]["content"]) for i in idxs]
    preds_n = [normalize(p) for p in preds]
    em = sum(1 for p, g in zip(preds_n, golds) if p == g) / len(idxs)
    f1 = sum(token_f1(p, g) for p, g in zip(preds, golds)) / len(idxs)
    return {"n": len(idxs), "EM": round(em * 100, 1), "F1": round(f1 * 100, 2),
            "preds": preds, "golds": golds}

res = {}
res["fused"] = run_model("fused", FUSED, list(range(len(samples))), 96)
res["base_30"] = run_model("base(30条对照)", BASE, list(range(30)), 300)

# BLEU / ROUGE-L（corpus 级）
try:
    import sacrebleu
    res["fused"]["BLEU"] = round(sacrebleu.corpus_bleu(
        [res["fused"]["preds"]], [[g] for g in res["fused"]["golds"]]).score, 2)
except Exception as e:
    res["fused"]["BLEU"] = f"n/a ({e})"
try:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = [scorer.score(g, p) for p, g in zip(res["fused"]["preds"], res["fused"]["golds"])]
    res["fused"]["ROUGE_L"] = round(sum(s.rougeL.fmeasure for s in scores) / len(scores) * 100, 2)
except Exception as e:
    res["fused"]["ROUGE_L"] = f"n/a ({e})"

out = "/Users/Admin/Desktop/微调和强化学习/gemma4-lora-pilot/docvqa_eval_results.json"
json.dump(res, open(out, "w"), ensure_ascii=False, indent=2)

print("\n" + "=" * 40)
print(f"fused  360 条: EM={res['fused']['EM']}%  F1={res['fused']['F1']}%  "
      f"BLEU={res['fused'].get('BLEU')}  ROUGE_L={res['fused'].get('ROUGE_L')}")
print(f"base   30 条: EM={res['base_30']['EM']}%  F1={res['base_30']['F1']}%")
print(f"\n--- fused 前 8 条 预测 vs 标准答案 ---")
for p, g in zip(res["fused"]["preds"][:8], res["fused"]["golds"][:8]):
    print(f"  P: {p[:60]}\n  G: {g[:60]}")
print(f"\n--- base 前 4 条（300 token 仍无直接答案？）---")
for p in res["base_30"]["preds"][:4]:
    print(f"  {p[:80]!r}... {p[-60:]!r}")
print(f"结果存 {out}")
