"""A/B 通用能力对比：底座 vs 全量 LoRA 融合模型
1) 通用对话 loss（40 条）—— 量化退化程度
2) 6 道通用题生成对比 —— 定性看风格
"""
import json, time, math
import mlx.core as mx
from mlx.nn.losses import cross_entropy
from mlx_lm import load, generate as lm_generate

BASE = "/Users/Admin/.lmstudio/models/lmstudio-community/gemma-4-26B-A4B-it-QAT-MLX-4bit"
FUSED = "/Users/Admin/.lmstudio/models/local/gemma-4-26B-A4B-docvqa-lora-full"
EVAL = "/Users/Admin/Desktop/微调和强化学习/gemma4-lora-pilot/generic_eval_data.jsonl"

GEN_PROMPTS = [
    "写一首关于秋天的短诗，四句以内",
    "用一句话解释什么是大模型的微调",
    "推荐三首适合雨天听的歌，并说说理由",
    "一个水池有两个进水管，甲管单开6小时灌满，乙管单开4小时灌满，两管齐开几小时灌满？",
    "用Python写一个函数，统计列表里每个元素出现的次数",
    "我最近总是加班到很晚，感觉很累，你给点生活上的建议",
]

results = {}

for name, path in [("base", BASE), ("fused", FUSED)]:
    t0 = time.time()
    print(f"=== 加载 {name} ...", flush=True)
    model, tokenizer = load(path)
    mx.eval(model)
    print(f"=== {name} 加载完成 ({time.time()-t0:.0f}s)，评估通用 loss", flush=True)

    samples = [json.loads(l) for l in open(EVAL) if l.strip()]
    total_loss, total_tok = 0.0, 0
    for s in samples:
        msgs = s["messages"]
        prompt_ids = tokenizer.apply_chat_template(msgs[:-1], add_generation_prompt=True, tokenize=True)
        full_ids = tokenizer.apply_chat_template(msgs, tokenize=True)[:1024]
        if len(full_ids) <= len(prompt_ids):
            continue
        logits = model(mx.array([full_ids]))[0]  # (L, V)
        # 只在 assistant 答案部分算 loss（mask prompt）
        start = max(len(prompt_ids) - 1, 0)
        logit_s = logits[start:-1]
        target_s = mx.array(full_ids[start + 1:])
        ce = cross_entropy(logit_s, target_s, reduction="mean")
        total_loss += float(ce) * (len(target_s))
        total_tok += len(target_s)
    loss = total_loss / total_tok
    ppl = math.exp(loss)
    print(f"=== {name} 通用 loss = {loss:.4f}, PPL = {ppl:.3f} ({total_tok} tok / {len(samples)} 条)", flush=True)

    print(f"=== {name} 生成 6 道通用题", flush=True)
    gen_outputs = []
    for p in GEN_PROMPTS:
        pid = tokenizer.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=True)
        out = lm_generate(model, tokenizer, prompt=pid, max_tokens=96, verbose=False)
        gen_outputs.append({"prompt": p, "response": out.strip()})
        print(f"    ok: {p[:14]}", flush=True)

    results[name] = {"generic_loss": loss, "generic_ppl": ppl, "generations": gen_outputs}
    del model, tokenizer
    mx.clear_cache()

out_path = "/Users/Admin/Desktop/微调和强化学习/gemma4-lora-pilot/ab_test_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"=== 完成，结果存 {out_path}", flush=True)
