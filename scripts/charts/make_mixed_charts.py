#!/usr/bin/env python3
"""解析 mixed_train.log -> mixed_loss_data.json + mixed_loss_chart.html（完整图表：坐标轴/刻度/图例/标记）"""
import json, re, os, math
BASE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(BASE, "..", "mixed_train.log")
DATA_OUT = os.path.join(BASE, "..", "mixed_loss_data.json")
HTML_OUT = os.path.join(BASE, "..", "mixed_loss_chart.html")

# ---------- 解析 ----------
ansi = re.compile(r"\x1b\[[0-9;]*m")
val, tr, tok, saves = [], [], [], []
for line in open(LOG, encoding="utf-8"):
    line = ansi.sub("", line).strip()
    m = re.match(r"(\d+)\s+val\s+([\d.]+)", line)
    if m: val.append([int(m.group(1)), float(m.group(2))])
    m = re.match(r"(\d+)\s+([\d.]+)\s+[▼▲]\s+(\d+)\s+([\d.k]+)", line)
    if m: tr.append([int(m.group(1)), float(m.group(2))]); tok.append([int(m.group(1)), int(m.group(3))])
    m = re.match(r"save\s+(\S+)", line)
    if m: saves.append(m.group(1))

best = min(val, key=lambda x: x[1])
data = {"train": tr, "val": val, "toks": tok, "saves": saves, "best_val": best,
        "crash_step": 7670, "note": "7670步 macOS GPU 抢占崩溃；选用 7500 ckpt fuse"}
json.dump(data, open(DATA_OUT, "w"))
print(f"val {len(val)} | train {len(tr)} | best {best[1]}@{best[0]}")

# ---------- 绘图工具 ----------
def nice_ticks(vmin, vmax, target=6):
    span = vmax - vmin
    if span <= 0: span = 1.0
    step = span / max(target - 1, 1)
    mag = 10 ** math.floor(math.log10(step))
    for r in (1, 2, 5, 10):
        if r * mag >= step:
            step = r * mag
            break
    start = math.floor(vmin / step) * step
    ticks, v = [], start
    while v <= vmax + step * 0.5:
        ticks.append(round(v, 8)); v += step
    return ticks, start, ticks[-1]

def chart(series, w=920, h=310, xl=58, xr=18, xt=30, xb=46,
          marks=None, legend=None, xfmt="{:d}", yfmt="{:.2f}",
          xtitle="训练步数", ytitle=""):
    """series: [{pts, color, label, dash}] -> 完整 SVG（网格/轴/刻度/图例/标记）
    颜色全部带 CSS 变量回退值，脱离 App 主题也能正常显示"""
    FG = "var(--foreground,#1d1d1f)"; MUT = "var(--muted-foreground,#6e6e73)"; BRD = "var(--border,#d2d2d7)"
    allp = [p for s in series for p in s["pts"]]
    xs = [p[0] for p in allp]; ys = [p[1] for p in allp]
    yt, y0, y1 = nice_ticks(min(ys), max(ys))
    xt_, x0, x1 = nice_ticks(min(xs), max(xs), 7)
    plotw, ph = w - xl - xr, h - xt - xb
    def X(x): return xl + (x - x0) / (x1 - x0) * plotw if x1 > x0 else xl
    def Y(y): return xt + (1 - (y - y0) / (y1 - y0)) * ph if y1 > y0 else xt
    out = []
    # 网格 + y 刻度数字
    for t in yt:
        out.append(f'<line x1="{xl}" y1="{Y(t):.1f}" x2="{w-xr}" y2="{Y(t):.1f}" stroke="var(--border,#d2d2d7)" stroke-width="0.6" opacity="0.5"/>')
        out.append(f'<text x="{xl-8}" y="{Y(t)+4:.1f}" text-anchor="end" font-size="12" fill="var(--muted-foreground,#6e6e73)">{yfmt.format(t)}</text>')
    # x 网格 + 刻度数字
    for t in xt_:
        out.append(f'<line x1="{X(t):.1f}" y1="{xt}" x2="{X(t):.1f}" y2="{h-xb}" stroke="var(--border,#d2d2d7)" stroke-width="0.4" opacity="0.25"/>')
        out.append(f'<text x="{X(t):.1f}" y="{h-xb+18}" text-anchor="middle" font-size="12" fill="var(--muted-foreground,#6e6e73)">{xfmt.format(t)}</text>')
    # 坐标轴
    out.append(f'<line x1="{xl}" y1="{h-xb}" x2="{w-xr}" y2="{h-xb}" stroke="var(--foreground,#1d1d1f)" stroke-width="1"/>')
    out.append(f'<line x1="{xl}" y1="{xt}" x2="{xl}" y2="{h-xb}" stroke="var(--foreground,#1d1d1f)" stroke-width="1"/>')
    out.append(f'<text x="{xl+plotw/2}" y="{h-6}" text-anchor="middle" font-size="12" fill="var(--muted-foreground,#6e6e73)">{xtitle}</text>')
    if ytitle:
        out.append(f'<text x="14" y="{xt+ph/2}" text-anchor="middle" font-size="12" fill="var(--muted-foreground,#6e6e73)" transform="rotate(-90 14 {xt+ph/2})">{ytitle}</text>')
    # 曲线
    for s in series:
        pts_s = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in s["pts"])
        out.append(f'<polyline points="{pts_s}" fill="none" stroke="{s["color"]}" stroke-width="2.2" {s.get("dash","")}/>')
    # 标记（竖线 / 圆点）
    for mk in (marks or []):
        x = X(mk["x"])
        if mk.get("type") == "vline":
            out.append(f'<line x1="{x:.0f}" y1="{xt}" x2="{x:.0f}" y2="{h-xb}" stroke="{mk["color"]}" stroke-width="1.5" stroke-dasharray="5 3"/>')
            out.append(f'<text x="{x+5:.0f}" y="{xt+14}" font-size="11.5" fill="{mk["color"]}">{mk["label"]}</text>')
        else:
            yy = Y(mk["y"])
            out.append(f'<circle cx="{x:.1f}" cy="{yy:.1f}" r="5" fill="{mk["color"]}" stroke="var(--card)" stroke-width="1.5"/>')
            dy = -12 if mk.get("above", True) else 20
            out.append(f'<text x="{x:.1f}" y="{yy+dy}" text-anchor="middle" font-size="11.5" fill="{mk["color"]}">{mk["label"]}</text>')
    # 图例
    if legend:
        ly = xt + 14
        out.append(f'<rect x="{xl+8}" y="{ly-14}" width="{150*len(legend)+16}" height="26" rx="5" fill="var(--card,#ffffff)" opacity="0.94" stroke="var(--border,#d2d2d7)"/>')
        lx = xl + 18
        for label, color, dash in legend:
            out.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+26}" y2="{ly}" stroke="{color}" stroke-width="2.5" {dash}/>')
            out.append(f'<text x="{lx+32}" y="{ly+4}" font-size="12" fill="var(--foreground,#1d1d1f)">{label}</text>')
            lx += 32 + len(label) * 12.5 + 46
    return "".join(out)

def card(title, svg, w=920, h=310):
    return f'<div class="card"><h3>{title}</h3><svg viewBox="0 0 {w} {h}" width="100%">{svg}</svg></div>'

# ---------- 各面板 ----------
# step-1 单点评估的冷启动尖峰（mixed 4.22 / full 8.91）会把平台区压扁，绘图时滤掉
vpts = [p for p in val if p[0] > 1]
full_val = [p for p in json.load(open(os.path.join(BASE, "..", "full_loss_data.json")))["val"] if p[0] > 1]

# ① val（崩溃竖线 + 7500 选用点）
p1 = card("① 验证集 Loss（每 100 步评估一次，750 条混合 val 集）",
    chart([{"pts": vpts, "color": "#e8b44a", "label": "val"}],
          marks=[{"type": "vline", "x": 7670, "color": "#e05c5c", "label": "7670 步 macOS GPU 抢占崩溃"},
                 {"x": 7500, "y": 0.447, "color": "#5aa86a", "label": "7500 ckpt（选用）", "above": False}],
          legend=[("val loss（本次混合跑）", "#e8b44a", "")],
          ytitle="val loss", xtitle="训练步数"))

# ② train
p2 = card("② 训练 Loss（每 100 步记录）",
    chart([{"pts": tr, "color": "#5c8ce0"}],
          legend=[("train loss", "#5c8ce0", "")],
          ytitle="train loss", xtitle="训练步数"))

# ③ tok/s
p3 = card("③ 吞吐量 tok/s（每 100 步）",
    chart([{"pts": tok, "color": "#9a7bd8"}],
          legend=[("tok/s", "#9a7bd8", "")], yfmt="{:.0f}",
          ytitle="吞吐 (tok/s)", xtitle="训练步数"))

# ④ 对比：并排双子图（两个 val 集水平差太远，共轴会看成"两条带"→ 各用各的坐标系）
try:
    full_best = min(full_val, key=lambda p: p[1])
    h4a = chart([{"pts": vpts, "color": "#e8b44a"}], w=440, h=300,
                marks=[{"x": 7500, "y": 0.447, "color": "#5aa86a", "label": "7500 ckpt（选用）", "above": False}],
                legend=[("本次混合跑 val", "#e8b44a", "")], ytitle="val loss")
    h4b = chart([{"pts": full_val, "color": "#8f9aa8"}], w=440, h=300,
                marks=[{"x": full_best[0], "y": full_best[1], "color": "#5aa86a", "label": f"best {full_best[1]:.2f}@step{full_best[0]}", "above": True}],
                legend=[("上次全量跑 val", "#8f9aa8", "")], ytitle="val loss")
    p4 = f'''<div class="card"><h3>④ val 曲线对比（两个 val 集水平不同，各自坐标系，看收敛形态）</h3>
<div class="duo">
<div><h4>本次混合跑 · 7,670 步（750 条混合 val）</h4><svg viewBox="0 0 440 300" width="100%">{h4a}</svg></div>
<div><h4>上次纯 DocVQA 全量跑 · 15,000 步（750 条 DocVQA val）</h4><svg viewBox="0 0 440 300" width="100%">{h4b}</svg></div>
</div></div>'''
except Exception as e:
    p4 = f"<div class='card'><h3>④ 对比</h3><p>缺 full_loss_data.json: {e}</p></div>"

# ⑤ 数据配比（条形图 + 数值/来源标注）
rows = [("DocVQA", 70, 10500, "#e8b44a", "上次筛选优质集抽样"),
        ("代码", 16, 2400, "#5c8ce0", "魔塔 CodeExercise-Python-27k"),
        ("文档", 4, 600, "#5aa86a", "魔塔 HUST-Student-Handbook QA"),
        ("对话", 10, 1500, "#9a7bd8", "手写高质量多风格样本")]
BW, BH, bl, bt = 920, 200, 150, 24
maxw = BW - bl - 340
bars = [f'<line x1="{bl}" y1="{bt-8}" x2="{bl}" y2="{BH-8}" stroke="var(--border,#d2d2d7)" stroke-width="0.6"/>']
for i, (name, pct, n, color, src) in enumerate(rows):
    y = bt + i * 44
    bw = max(10, maxw * pct / 100)
    bars.append(f'<text x="{bl-10}" y="{y+20}" text-anchor="end" font-size="13" fill="var(--foreground,#1d1d1f)">{name} {pct}%</text>')
    bars.append(f'<rect x="{bl}" y="{y}" width="{bw:.0f}" height="30" fill="{color}" rx="4"/>')
    bars.append(f'<text x="{bl+bw+10}" y="{y+20}" font-size="12.5" fill="var(--muted-foreground,#6e6e73)">= {n:,} 条 · {src}</text>')
p5 = f'''<div class="card"><h3>⑤ 数据配比（共 15,000 条；train/valid/test = 13,500 / 750 / 750，各 split 同比例）</h3>
<svg viewBox="0 0 {BW} {BH}" width="100%">{"".join(bars)}</svg></div>'''

# ---------- 页面 ----------
html = f'''<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>Gemma4 混合数据 LoRA 训练曲线（完整图表）</title>
<style>
body{{font-family:-apple-system,'PingFang SC',sans-serif;max-width:960px;margin:24px auto;padding:0 16px;color:var(--foreground,#1d1d1f);background:transparent}}
h1{{font-size:21px}} h2{{font-size:13.5px;opacity:.72;font-weight:400;line-height:1.55}} h3{{font-size:13.5px;margin:0 0 10px;opacity:.9}} h4{{font-size:12.5px;margin:0 0 6px;opacity:.75;font-weight:500}}
.card{{background:var(--card,#ffffff);border:1px solid var(--border,#d2d2d7);border-radius:10px;padding:14px 16px;margin:14px 0}}
.duo{{display:flex;gap:18px}} .duo>div{{flex:1;min-width:0}}
.meta{{font-size:13px;opacity:.85;line-height:1.85}}
</style></head><body>
<h1>Gemma 4 26B-A4B · 混合数据 LoRA 训练曲线</h1>
<h2>70% DocVQA + 20% 代码/文档 + 10% 对话 · 计划 13,500 步 · 实际 7,670 步（57%）因 macOS GPU 抢占崩溃 · 选用 <b>7500 步 ckpt</b>（val≈0.447，平台区最佳）fuse 为最终模型<br>
注：绘图已滤除第 1 步冷启动单点（mixed val 4.22 / full val 8.91），平台区看得更清楚</h2>
{p1}
{p2}
{p3}
{p4}
{p5}
<div class="card meta"><b>关键数字</b><br>
· 实际 7,670/13,500 步（57%），用时 ~6.5h，平均 ~3.0s/步，吞吐 ~40 tok/s<br>
· 最佳 val <b>{best[1]} @ step {best[0]}</b>；val 自 ~3000 步起进入 0.37–0.62 平台（无过拟合）<br>
· train loss 2.9 → ~1.3 持续下降 · ckpt 2500/5000/7500 → 选用 7500 fuse 成 gemma-4-26B-A4B-mixed-lora（14.47G）</div>
</body></html>'''
open(HTML_OUT, "w").write(html)
print("已写:", HTML_OUT, os.path.getsize(HTML_OUT), "bytes")
