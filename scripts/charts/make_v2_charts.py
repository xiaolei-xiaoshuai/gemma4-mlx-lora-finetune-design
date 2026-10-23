#!/usr/bin/env python3
"""解析 mixed_v2_train.log + mixed_v2_resume.log -> mixed_v2_loss_chart.html
拼接规则：首跑 0-7500（崩溃于 8600 但未用段）+ 续跑 7500-13498（相对步 +7500）
"""
import json, re, os, math
BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
HTML_OUT = os.path.join(ROOT, "mixed_v2_loss_chart.html")

ansi = re.compile(r"\x1b\[[0-9;]*m")

def parse(logpath):
    val, tr, tok, saves = [], [], [], []
    for line in open(logpath, encoding="utf-8"):
        line = ansi.sub("", line).strip()
        m = re.match(r"(\d+)\s+val\s+([\d.]+)", line)
        if m: val.append([int(m.group(1)), float(m.group(2))])
        m = re.match(r"(\d+)\s+([\d.]+)\s+[▼▲]\s+(\d+)\s+([\d.k]+)", line)
        if m: tr.append([int(m.group(1)), float(m.group(2))]); tok.append([int(m.group(1)), int(m.group(3))])
        m = re.match(r"save\s+(\S+)", line)
        if m: saves.append(m.group(1))
    return val, tr, tok, saves

# 首跑 0-8600（用 0-7500 段）+ 续跑（相对步+7500）
v1, t1, k1, s1 = parse(os.path.join(ROOT, "mixed_v2_train.log"))
v2, t2, k2, s2 = parse(os.path.join(ROOT, "mixed_v2_resume.log"))
OFF = 7500
def shift(arr, off): return [[a+off, b] for a, b in arr]
val = [p for p in v1 if p[0] <= OFF] + [p for p in v2 if p[0] > 0]
val = [p for p in val]  # v2 续跑段已 +7500
val = [p for p in ([p for p in v1 if p[0] <= OFF] + shift(v2, OFF))]
val = [p for p in val if p[0] > 1]
tr = [p for p in ([p for p in t1 if p[0] <= OFF] + shift(t2, OFF))]
tok = [p for p in ([p for p in k1 if p[0] <= OFF] + shift(k2, OFF))]
saves = s1 + s2
best = min(val, key=lambda x: x[1])
print(f"val {len(val)} 点 (0-13498) | train {len(tr)} | best {best[1]}@{best[0]}")

# ---------- 绘图工具（同 v1，带 CSS 回退） ----------
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
    FG = "var(--foreground,#1d1d1f)"; MUT = "var(--muted-foreground,#6e6e73)"; BRD = "var(--border,#d2d2d7)"
    allp = [p for s in series for p in s["pts"]]
    xs = [p[0] for p in allp]; ys = [p[1] for p in allp]
    yt, y0, y1 = nice_ticks(min(ys), max(ys))
    xt_, x0, x1 = nice_ticks(min(xs), max(xs), 7)
    plotw, ph = w - xl - xr, h - xt - xb
    def X(x): return xl + (x - x0) / (x1 - x0) * plotw if x1 > x0 else xl
    def Y(y): return xt + (1 - (y - y0) / (y1 - y0)) * ph if y1 > y0 else xt
    out = []
    for t in yt:
        out.append(f'<line x1="{xl}" y1="{Y(t):.1f}" x2="{w-xr}" y2="{Y(t):.1f}" stroke="var(--border,#d2d2d7)" stroke-width="0.6" opacity="0.5"/>')
        out.append(f'<text x="{xl-8}" y="{Y(t)+4:.1f}" text-anchor="end" font-size="12" fill="var(--muted-foreground,#6e6e73)">{yfmt.format(t)}</text>')
    for t in xt_:
        out.append(f'<line x1="{X(t):.1f}" y1="{xt}" x2="{X(t):.1f}" y2="{h-xb}" stroke="var(--border,#d2d2d7)" stroke-width="0.4" opacity="0.25"/>')
        out.append(f'<text x="{X(t):.1f}" y="{h-xb+18}" text-anchor="middle" font-size="12" fill="var(--muted-foreground,#6e6e73)">{xfmt.format(t)}</text>')
    out.append(f'<line x1="{xl}" y1="{h-xb}" x2="{w-xr}" y2="{h-xb}" stroke="var(--foreground,#1d1d1f)" stroke-width="1"/>')
    out.append(f'<line x1="{xl}" y1="{xt}" x2="{xl}" y2="{h-xb}" stroke="var(--foreground,#1d1d1f)" stroke-width="1"/>')
    out.append(f'<text x="{xl+plotw/2}" y="{h-6}" text-anchor="middle" font-size="12" fill="var(--muted-foreground,#6e6e73)">{xtitle}</text>')
    if ytitle:
        out.append(f'<text x="14" y="{xt+ph/2}" text-anchor="middle" font-size="12" fill="var(--muted-foreground,#6e6e73)" transform="rotate(-90 14 {xt+ph/2})">{ytitle}</text>')
    for s in series:
        pts_s = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in s["pts"])
        out.append(f'<polyline points="{pts_s}" fill="none" stroke="{s["color"]}" stroke-width="2.2" {s.get("dash","")}/>')
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

# ① val（8600 崩溃竖线 + 7500 续训点）
p1 = card("① 验证集 Loss（每 100 步，751 条混合 val；8600 步崩溃后从 7500 ckpt 续跑）",
    chart([{"pts": val, "color": "#e8b44a", "label": "val"}],
          marks=[{"type": "vline", "x": 8600, "color": "#e05c5c", "label": "8600 崩溃 → 7500 续训"},
                 {"x": 7500, "y": 0.383, "color": "#5aa86a", "label": "7500 ckpt（接续点）", "above": False}],
          legend=[("v2 val loss", "#e8b44a", "")],
          ytitle="val loss", xtitle="训练步数"))

# ② train
p2 = card("② 训练 Loss（每 100 步）",
    chart([{"pts": tr, "color": "#5c8ce0"}],
          legend=[("train loss", "#5c8ce0", "")],
          ytitle="train loss", xtitle="训练步数"))

# ③ tok/s
p3 = card("③ 吞吐量 tok/s（每 100 步）",
    chart([{"pts": tok, "color": "#9a7bd8"}],
          legend=[("tok/s", "#9a7bd8", "")], yfmt="{:.0f}",
          ytitle="吞吐 (tok/s)", xtitle="训练步数"))

# ④ 对比 v1（并排双子图，各自坐标系）
try:
    v1data = json.load(open(os.path.join(BASE, "..", "mixed_loss_data.json")))
    v1val = [p for p in v1data["val"] if p[0] > 1]
    v1best = min(v1val, key=lambda p: p[1])
    h4a = chart([{"pts": val, "color": "#e8b44a"}], w=440, h=300,
                marks=[{"x": 8600, "y": 0.40, "color": "#e05c5c", "label": "崩溃", "above": False}],
                legend=[("v2 混合跑 val", "#e8b44a", "")], ytitle="val loss")
    h4b = chart([{"pts": v1val, "color": "#8f9aa8"}], w=440, h=300,
                marks=[{"x": v1best[0], "y": v1best[1], "color": "#5aa86a", "label": f"best {v1best[1]:.2f}@{v1best[0]}", "above": True}],
                legend=[("v1 混合跑 val（灰）", "#8f9aa8", "")], ytitle="val loss")
    p4 = f'''<div class="card"><h3>④ val 曲线对比：v2 混合跑 vs v1 混合跑（val 集不同，各自坐标系，看形态/水平）</h3>
<div class="duo">
<div><h4>v2 · 50/25/25 · 13,498 步（751 条混合 val）</h4><svg viewBox="0 0 440 300" width="100%">{h4a}</svg></div>
<div><h4>v1 · 70/16/4/10 · 7,670 步（750 条混合 val）</h4><svg viewBox="0 0 440 300" width="100%">{h4b}</svg></div>
</div></div>'''
except Exception as e:
    p4 = f"<div class='card'><h3>④ 对比</h3><p>缺 mixed_loss_data.json: {e}</p></div>"

# ⑤ 数据配比（50/25/25）
rows = [("DocVQA", 50, 7500, "#e8b44a", "上次筛选优质集 10,500 抽样"),
        ("代码", 25, 3750, "#5c8ce0", "CodeExercise-Python-27k 抽样"),
        ("通用对话", 25, 3750, "#9a7bd8", "gen_dialogue_v2 生成（v1 1500 + 新种子池）")]
BW, BH, bl, bt = 920, 200, 150, 24
maxw = BW - bl - 340
bars = [f'<line x1="{bl}" y1="{bt-8}" x2="{bl}" y2="{BH-8}" stroke="var(--border,#d2d2d7)" stroke-width="0.6"/>']
for i, (name, pct, n, color, src) in enumerate(rows):
    y = bt + i * 44
    bw = max(10, maxw * pct / 100)
    bars.append(f'<text x="{bl-10}" y="{y+20}" text-anchor="end" font-size="13" fill="var(--foreground,#1d1d1f)">{name} {pct}%</text>')
    bars.append(f'<rect x="{bl}" y="{y}" width="{bw:.0f}" height="30" fill="{color}" rx="4"/>')
    bars.append(f'<text x="{bl+bw+10}" y="{y+20}" font-size="12.5" fill="var(--muted-foreground,#6e6e73)">= {n:,} 条 · {src}</text>')
p5 = f'''<div class="card"><h3>⑤ 数据配比（共 15,000 条；train/valid/test = 13,498 / 751 / 751，各 split 同比例）</h3>
<svg viewBox="0 0 {BW} {BH}" width="100%">{" ".join(bars)}</svg></div>'''

html = f'''<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>Gemma4 v2 混合数据 LoRA 训练曲线</title>
<style>
body{{font-family:-apple-system,'PingFang SC',sans-serif;max-width:960px;margin:24px auto;padding:0 16px;color:var(--foreground,#1d1d1f);background:transparent}}
h1{{font-size:21px}} h2{{font-size:13.5px;opacity:.72;font-weight:400;line-height:1.55}} h3{{font-size:13.5px;margin:0 0 10px;opacity:.9}} h4{{font-size:12.5px;margin:0 0 6px;opacity:.75;font-weight:500}}
.card{{background:var(--card,#ffffff);border:1px solid var(--border,#d2d2d7);border-radius:10px;padding:14px 16px;margin:14px 0}}
.duo{{display:flex;gap:18px}} .duo>div{{flex:1;min-width:0}}
.meta{{font-size:13px;opacity:.85;line-height:1.85}}
</style></head><body>
<h1>Gemma 4 26B-A4B · v2 混合数据 LoRA 训练曲线</h1>
<h2>50% DocVQA + 25% 代码 + 25% 对话 · 计划 13,498 步 · <b>实际跑满 13,498 步（100%）</b>（中途 8,600 步 macOS GPU 崩溃，从 7,500 ckpt 续跑收尾）<br>
注：首跑 0–7,500 段与续跑 7,500–13,498 段拼接；已滤除第 1 步冷启动单点</h2>
{p1}
{p2}
{p3}
{p4}
{p5}
<div class="card meta"><b>关键数字</b><br>
· 实际 <b>13,498/13,498 步（100%）</b>：首跑 7.5h 崩于 8,600 → 从 7,500 ckpt 续跑 ~6h 完成<br>
· 最佳 val <b>{best[1]:.3f} @ step {best[0]}</b>；val 自 ~3,500 步起进入 0.35–0.48 平台（无过拟合）<br>
· 收尾 test loss 0.428 / ppl 1.535 · ckpt 2500/5000/7500 + 续跑 9000/10500/12000 + 最终 13,498<br>
· 选用 <b>13,498 步最终 adapter</b> fuse 成 gemma-4-26B-A4B-mixed-v2-lora（14G）</div>
</body></html>'''
open(HTML_OUT, "w").write(html)
print("已写:", HTML_OUT, os.path.getsize(HTML_OUT), "bytes")
