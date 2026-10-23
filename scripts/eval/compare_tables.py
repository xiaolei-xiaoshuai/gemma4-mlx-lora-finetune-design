#!/usr/bin/env python3
"""多模型 × 多组 对比表生成器。
用法:
  python compare_tables.py <results1.json:标签1> <results2.json:标签2> ... -o 对比表.md
  例:
  python compare_tables.py run_base.json:基座 run_v1.json:v1混合 run_v2.json:v2混合 run_full.json:100%DocVQA -o 对比表.md
  标签缺省 = 文件名。
输出: 一个 Markdown + 一个 HTML（同名 .html）
"""
import json, sys, os, html


def parse_args(argv):
    pairs, out = [], None
    i = 1
    while i < len(argv):
        a = argv[i]
        if a in ("-o", "--out") and i + 1 < len(argv):
            out = argv[i + 1]; i += 2; continue
        if ":" in a and os.path.exists(a.split(":", 1)[0]):
            path, label = a.split(":", 1); pairs.append((label, path))
        else:
            label = os.path.splitext(os.path.basename(a))[0]
            pairs.append((label, a))
        i += 1
    if not pairs:
        print("至少给一个 results.json[:标签]"); sys.exit(1)
    return pairs, out


def load(results_path):
    d = json.load(open(results_path, encoding="utf-8"))
    return d


def main():
    pairs, out = parse_args(sys.argv)
    if not out:
        out = "对比表.md"
    base_md = os.path.splitext(out)[0]
    results = [(label, load(p)) for label, p in pairs]

    # 收集所有组
    allgroups = []
    for _, d in results:
        for g in d.get("groups", {}):
            if g not in allgroups:
                allgroups.append(g)

    # 指标列
    lines = []
    lines.append("# 多模型 × 多领域对比表\n")
    lines.append("模型：" + " / ".join(l for l, _ in results) + "\n")

    # 表1: 组级汇总（thought 泄漏 / 代码围栏 / 题数）
    lines.append("\n## 一、组级汇总\n")
    lines.append("| 组 | " + " | ".join(f"{l} 泄漏 | {l} 围栏 | {l} 题数" for l, _ in results) + " |")
    lines.append("|" + "---|" * (1 + len(results) * 3))
    for g in allgroups:
        row = [g]
        for label, d in results:
            s = d.get("groups", {}).get(g, {}).get("summary", {})
            row += [s.get("thought_leak", "-"), s.get("code_fenced", "-"), s.get("n", "-")]
        lines.append("| " + " | ".join(str(x) for x in row) + " |")

    # 表2: 逐题明细（每个组一个小表）
    lines.append("\n## 二、逐题明细\n")
    for g in allgroups:
        lines.append(f"\n### 组 [{g}]\n")
        # 并排所有模型该组的问题
        per_model_q = {label: {q["id"]: q for q in d.get("groups", {}).get(g, {}).get("questions", [])}
                       for label, d in results}
        qids = []
        for label, _ in results:
            for qid in per_model_q[label]:
                if qid not in qids:
                    qids.append(qid)
        lines.append("| 题 | " + " | ".join(l for l, _ in results) + " |")
        lines.append("|" + "---|" * (1 + len(results)))
        for qid in qids:
            row = [qid]
            for label, _ in results:
                q = per_model_q[label].get(qid)
                if q:
                    r = q["response"].replace("\n", " ")
                    row.append(r[:120] + ("…" if len(r) > 120 else ""))
                else:
                    row.append("—")
            lines.append("| " + " | ".join(str(x) for x in row) + " |")

    md = "\n".join(lines)
    with open(base_md + ".md", "w", encoding="utf-8") as f:
        f.write(md)

    # 生成简单 HTML
    def md_table_to_html(md_text):
        # 极简: 把 Markdown 转成 HTML（保留表格 + 标题）
        import re
        h = []
        in_table = False
        row_count = 0
        for ln in md_text.splitlines():
            if ln.startswith("|"):
                if not in_table:
                    h.append("<table>"); in_table = True; row_count = 0
                cells = [c.strip() for c in ln.strip("|").split("|")]
                if all(set(c) <= set("-: ") and c != "" for c in cells):
                    continue
                tag = "th" if row_count == 0 else "td"
                row_count += 1
                h.append("<tr>" + "".join(f"<{tag}>{html.escape(c)}</{tag}>" for c in cells) + "</tr>")
            else:
                if in_table:
                    h.append("</table>"); in_table = False
                if ln.startswith("### "):
                    h.append(f"<h3>{html.escape(ln[4:])}</h3>")
                elif ln.startswith("## "):
                    h.append(f"<h2>{html.escape(ln[3:])}</h2>")
                elif ln.startswith("# "):
                    h.append(f"<h1>{html.escape(ln[2:])}</h1>")
                elif ln.strip():
                    h.append(f"<p>{html.escape(ln)}</p>")
        if in_table:
            h.append("</table>")
        return "\n".join(h)

    body = md_table_to_html(md)
    html_doc = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>多模型对比表</title>
<style>
 body{{max-width:1100px;margin:24px auto;font:15px/1.6 -apple-system,'PingFang SC',sans-serif;color:#1a1a1a;padding:0 16px}}
 table{{width:100%;border-collapse:collapse;margin:10px 0;font-size:13px}}
 th,td{{border:1px solid #ddd;padding:6px 8px;vertical-align:top;word-break:break-word}}
 th{{background:#f5f5f7}}
 h1{{font-size:24px}}h2{{font-size:19px;margin-top:28px}}h3{{font-size:16px;margin-top:18px}}
 td{{max-width:280px}}
</style></head><body>{body}</body></html>"""
    with open(base_md + ".html", "w", encoding="utf-8") as f:
        f.write(html_doc)

    print(f"已生成 {base_md}.md 和 {base_md}.html")
    print("组:", ", ".join(allgroups))


if __name__ == "__main__":
    main()
