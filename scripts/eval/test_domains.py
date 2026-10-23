#!/usr/bin/env python3
"""三领域测试：正确套 chat 模板 + 非贪心采样（temp0.7/top_p0.95）
用法: python test_domains.py <model_path> <out_json>
"""
import json, sys
import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

MODEL = sys.argv[1]
OUT = sys.argv[2]

# 固定题目（全部是新题，不在训练集）
Q_DOCVQA = """你是一名文档问答助手。根据下面的文档内容回答问题。

【文档】
3.4 贷款发放
3.4.1 贷款发放对象
具有中华人民共和国国籍或永久居留权，具有完全民事行为能力，在贷款银行开立个人结算账户的个人客户。
3.4.2 贷款发放条件
(1) 年龄18周岁（含）至65周岁（含）；(2) 具有合法、稳定的收入来源；(3) 信用记录良好；(4) 有明确的贷款用途；(5) 同意并遵守贷款银行的各项贷款规定。
3.4.3 贷款金额
个人贷款额度一般为1000元至50万元（含），具体额度由贷款银行根据借款人的资信状况和偿还能力核定。

【问题】申请个人贷款需要满足哪些条件？"""

Q_CODE = """用 Python 写一个函数 `moving_average(nums, window)`，返回列表的滑动平均值（四舍五入到小数点后2位）。
要求：
1. 当 len(nums) < window 时返回 []
2. 输入 [1,2,3,4,5]，window=2 时输出 [1.5, 2.5, 3.5, 4.5]
3. 不要使用统计库，手写累加。直接给出可运行的代码。"""

Q_DLG_1 = "我连续两周失眠，早上7点要上班，怎么办？"
Q_DLG_2 = "用三句话给我讲讲什么是通货膨胀。"
Q_DLG_3 = "帮我给好朋友写一段生日祝福，要幽默一点。"

QUERIES = [
    ("docvqa", Q_DOCVQA),
    ("code", Q_CODE),
    ("dlg1", Q_DLG_1),
    ("dlg2", Q_DLG_2),
    ("dlg3", Q_DLG_3),
]

print(f"加载模型 {MODEL} ...", flush=True)
model, tokenizer = load(MODEL)
print("模型加载完成", flush=True)

sampler = make_sampler(temp=0.7, top_p=0.95)

results = {}
for name, text in QUERIES:
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], add_generation_prompt=True
    )
    print(f"\n===== [{name}] 问题 =====\n{text[:120]}...", flush=True)
    out = generate(model, tokenizer, prompt, max_tokens=512, sampler=sampler, verbose=False)
    print(f"===== [{name}] 回答 =====\n{out}", flush=True)
    results[name] = out
    mx.metal.clear_cache()

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\n已保存 {OUT}")
