# 多领域测试框架（可扩展）

三件套 + 一个转换器，支持「任意模型 × 任意组」测试并自动生成对比表。

## 文件
| 文件 | 作用 |
|---|---|
| `testing/test_suite.json` | 问题库：按 `groups` 组织，每组 `{description, questions:[{id, prompt}]}`。`prompt` 为字符串=单轮，数组=多轮 |
| `scripts/eval/run_suite.py` | 运行器：加载模型，跑指定组，自动记录 `thought_leak / code_fenced / 长度 / 耗时` |
| `scripts/eval/compare_tables.py` | 对比表：N 个模型结果 → Markdown + HTML（组级汇总 + 逐题明细） |
| `scripts/eval/convert_legacy.py` | 把旧的扁平 `{题:回答}` JSON 转成新格式，方便和历史结果混比 |

## 跑一个模型
```bash
# 全部组
~/venvs/mlx/bin/python scripts/eval/run_suite.py <模型路径> out.json testing/test_suite.json
# 只跑几组（docvqa / code / math）
~/venvs/mlx/bin/python scripts/eval/run_suite.py <模型路径> out.json testing/test_suite.json docvqa code math
```

## 生成对比表
```bash
~/venvs/mlx/bin/python scripts/eval/compare_tables.py \
  out_base.json:基座 out_full.json:100%DocVQA out_v2.json:v2混合 \
  -o 对比表
# 生成 对比表.md + 对比表.html
```

## 加新组 / 新题
在 `test_suite.json` 的 `groups` 里加一个键即可，`run_suite.py` 与 `compare_tables.py` 会自动纳入：
```json
"nlp": {
  "description": "命名实体",
  "questions": [
    {"id": "nlp_1", "prompt": "从下面句子抽出人名、地点、机构：…"},
    {"id": "nlp_2", "prompt": [] }   // 多轮就传数组
  ]
}
```

## 现有 8 组
`docvqa`(3) · `code`(3) · `dialogue`(3) · `multi_turn`(2) · `math`(2) · `reasoning`(2) · `summarization`(2) · `instruction`(2)
全部为训练集外新题。

## 指标含义
- `thought_leak`：回答里是否出现 `<|channel>thought`/`<channel>`（基座特有，微调后应消失）
- `code_fenced`：是否用 ``` 围栏（代码组期望 true）
- `n_chars / n_tokens_approx / seconds`：长度与耗时
