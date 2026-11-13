#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""procgen —— 程序化图像生成原型（DSL → SVG → Chrome 渲染 → PNG）

设计目标：让语言模型用「语义」描述画面，而不是手写坐标。
分层：
  1. DSL      高层图元（skyline / gradient / sun / haze / grain …）
  2. Compiler 图元 → SVG（含 filter / mask / 渐变）
  3. Render   SVG → HTML → Chrome headless → PNG

用法：
  python3 procgen.py scene.json out.png
  python3 procgen.py --demo demo.png
"""

import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path

CHROME = os.environ.get("PROCGEN_CHROME", "chromium")
TMP = Path(os.environ.get("PROCGEN_TMP", str(Path.home() / ".cache" / "procgen")))


# ---------------------------------------------------------------- 图元

def lerp(a, b, t):
    return a + (b - a) * t


def hex2rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def rgb2hex(r, g, b):
    return "#%02x%02x%02x" % (int(max(0, min(255, r))),
                              int(max(0, min(255, g))),
                              int(max(0, min(255, b))))


def mix(c1, c2, t):
    a, b = hex2rgb(c1), hex2rgb(c2)
    return rgb2hex(*[lerp(a[i], b[i], t) for i in range(3)])


def shade(c, k):
    """k<1 变暗，k>1 变亮"""
    r, g, b = hex2rgb(c)
    return rgb2hex(r * k, g * k, b * k)


# ---------------------------------------------------------------- 编译器

class Compiler:
    def __init__(self, w=1920, h=1080, seed=42):
        self.w, self.h = w, h
        self.rng = random.Random(seed)
        self.defs = []
        self.body = []
        self._ids = 0

    def _id(self, prefix="p"):
        self._ids += 1
        return f"{prefix}{self._ids}"

    # ---- 基础 ----
    def gradient(self, c1, c2, x1=0, y1=0, x2=0, y2=1, name=None):
        gid = name or self._id("grad")
        self.defs.append(
            f'<linearGradient id="{gid}" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}">'
            f'<stop offset="0" stop-color="{c1}"/><stop offset="1" stop-color="{c2}"/>'
            f'</linearGradient>')
        return gid

    def rect(self, x, y, w, h, fill, opacity=1.0, extra=""):
        self.body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
                         f'fill="{fill}" opacity="{opacity}" {extra}/>')

    def poly(self, pts, fill, opacity=1.0, extra=""):
        d = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        self.body.append(f'<polygon points="{d}" fill="{fill}" opacity="{opacity}" {extra}/>')

    def circle(self, cx, cy, r, fill, opacity=1.0, extra=""):
        self.body.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" '
                         f'fill="{fill}" opacity="{opacity}" {extra}/>')

    def text(self, x, y, s, size=32, fill="#fff", opacity=1.0, weight=400, family="Noto Sans CJK SC"):
        esc = (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
        self.body.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
                         f'opacity="{opacity}" font-weight="{weight}" '
                         f'font-family="{family}">{esc}</text>')

    # ---- 语义图元 ----
    def sky(self, top="#1e3a5f", bottom="#f2c98a", horizon=0.62):
        gid = self.gradient(top, bottom, 0, 0, 0, horizon * 1.15)
        self.rect(0, 0, self.w, horizon * self.h + 2, f"url(#{gid})")

    def sun(self, x=0.72, y=0.42, r=0.06, color="#fff3c4", glow=2.6):
        cx, cy, rad = x * self.w, y * self.h, r * self.h
        gid = self._id("glow")
        self.defs.append(
            f'<radialGradient id="{gid}">'
            f'<stop offset="0" stop-color="{color}" stop-opacity="0.95"/>'
            f'<stop offset="0.35" stop-color="{color}" stop-opacity="0.45"/>'
            f'<stop offset="1" stop-color="{color}" stop-opacity="0"/>'
            f'</radialGradient>')
        self.circle(cx, cy, rad * glow, f"url(#{gid})")
        self.circle(cx, cy, rad * 0.55, color, 0.9)

    def haze(self, y=0.55, height=0.18, color="#ffffff", opacity=0.35):
        gid = self._id("haze")
        self.defs.append(
            f'<linearGradient id="{gid}" x1="0" y1="0" x2="0" y2="1">'
            f'<stop offset="0" stop-color="{color}" stop-opacity="0"/>'
            f'<stop offset="0.5" stop-color="{color}" stop-opacity="{opacity}"/>'
            f'<stop offset="1" stop-color="{color}" stop-opacity="0"/>'
            f'</linearGradient>')
        self.rect(0, (y - height / 2) * self.h, self.w, height * self.h, f"url(#{gid})")

    def skyline(self, horizon=0.62, density=0.7, depth=3, base_color="#2a3b4d",
                window_p=0.35, seed=None, height_range=(0.08, 0.34), crowd=1.0):
        """多层天际线：越远越淡、越矮、越密（大气透视）"""
        rng = random.Random(seed if seed is not None else self.rng.random())
        for layer in range(depth):
            t = layer / max(1, depth - 1)          # 0=最远 1=最近
            color = mix(mix(base_color, "#cfe0ee", 0.72 * (1 - t)), base_color, 0.25 * t)
            op = 0.55 + 0.45 * t
            hy = (horizon - 0.02 * (1 - t)) * self.h
            hmin = height_range[0] * self.h * (0.45 + 0.55 * t)
            hmax = height_range[1] * self.h * (0.45 + 0.55 * t)
            x = -60.0
            while x < self.w + 60:
                bw = rng.uniform(28, 120) * (1.15 - 0.3 * t) * crowd
                gap = rng.uniform(2, 22) * (1.4 - 0.4 * t)
                bh = rng.uniform(hmin, hmax)
                # 建筑主体
                self.rect(x, hy - bh, bw, bh + 4, color, op)
                # 顶部细节（部分楼有天线/退台）
                if rng.random() < 0.25:
                    nw = bw * rng.uniform(0.3, 0.55)
                    nh = bh * rng.uniform(0.06, 0.16)
                    self.rect(x + (bw - nw) / 2, hy - bh - nh, nw, nh + 2, color, op)
                # 窗户（只在中近景画）
                if t > 0.25 and window_p > 0 and rng.random() < window_p:
                    self._windows(x, hy - bh, bw, bh, t, op)
                x += bw + gap

    def _windows(self, x, y, w, h, t, op):
        rng = self.rng
        wc = rgb2hex(*[rng.randint(150, 255) for _ in range(3)])
        cols = max(1, int(w / rng.uniform(14, 24)))
        rows = max(1, int(h / rng.uniform(16, 30)))
        pw, ph = w / cols * 0.5, h / rows * 0.35
        for r in range(rows):
            for c in range(cols):
                if rng.random() < 0.42 * t:
                    self.rect(x + c * (w / cols) + pw * 0.5,
                              y + r * (h / rows) + ph * 0.6,
                              pw, ph, wc, 0.55 * op)

    def ground(self, y=0.62, color="#1a1f26", texture=True, seed=None):
        rng = random.Random(seed if seed is not None else self.rng.random())
        gid = self.gradient(shade(color, 1.35), color, 0, y, 0, 1)
        self.rect(0, y * self.h, self.w, (1 - y) * self.h + 2, f"url(#{gid})")
        if texture:
            for _ in range(160):
                gx = rng.uniform(0, self.w)
                gy = rng.uniform(y * self.h, self.h)
                gw = rng.uniform(20, 160)
                gh = rng.uniform(1, 3)
                self.rect(gx, gy, gw, gh, shade(color, 1.6), rng.uniform(0.03, 0.10))

    def grain(self, amount=0.055, scale=1.0, seed=7):
        """胶片颗粒：用 feTurbulence 做，渲染快且真实"""
        fid = self._id("grain")
        self.defs.append(
            f'<filter id="{fid}" x="0" y="0" width="100%" height="100%">'
            f'<feTurbulence type="fractalNoise" baseFrequency="{0.9 * scale:.3f}" '
            f'numOctaves="3" seed="{seed}" result="n"/>'
            f'<feColorMatrix in="n" type="saturate" values="0"/>'
            f'</filter>')
        self.body.append(
            f'<rect width="{self.w}" height="{self.h}" filter="url(#{fid})" '
            f'opacity="{amount}" style="mix-blend-mode:overlay"/>')

    def vignette(self, strength=0.35):
        gid = self._id("vig")
        self.defs.append(
            f'<radialGradient id="{gid}" cx="0.5" cy="0.5" r="0.75">'
            f'<stop offset="0.55" stop-color="#000" stop-opacity="0"/>'
            f'<stop offset="1" stop-color="#000" stop-opacity="{strength}"/>'
            f'</radialGradient>')
        self.rect(0, 0, self.w, self.h, f"url(#{gid})")

    def blur_layer(self, blur_px=6):
        """景深：给已有图层加高斯模糊（用包裹 group 实现）"""
        if not self.body:
            return
        gid = self._id("blur")
        self.defs.append(f'<filter id="{gid}"><feGaussianBlur stdDeviation="{blur_px}"/></filter>')
        merged = "\n".join(self.body)
        self.body = [f'<g filter="url(#{gid})">{merged}</g>']


    def landmark(self, x=0.62, horizon=0.72, h=0.46, w=0.030, color="#2a3b4d",
                 taper=True, antenna=0.06, floors=26):
        """地标塔（如平安金融中心）：上窄下宽 + 天线"""
        cx = x * self.w
        top = (horizon - h) * self.h
        base = horizon * self.h
        bw = w * self.w
        if taper:
            pts = [(cx - bw / 2, base), (cx - bw * 0.28, top),
                   (cx + bw * 0.28, top), (cx + bw / 2, base)]
            self.poly(pts, color, 0.97)
        else:
            self.rect(cx - bw / 2, top, bw, base - top, color, 0.97)
        if antenna > 0:
            aw = max(2.0, bw * 0.06)
            self.rect(cx - aw / 2, top - antenna * self.h, aw, antenna * self.h + 2, color, 0.95)
        # 楼层横线
        for i in range(1, floors):
            fy = top + (base - top) * i / floors
            self.rect(cx - bw * 0.26, fy, bw * 0.52, max(0.8, (base - top) / floors * 0.10),
                      shade(color, 1.35), 0.25)

    def trees(self, y=0.80, count=9, size=(0.05, 0.11), color="#0d1418", seed=None,
              spread=(0.0, 1.0)):
        rng = random.Random(seed if seed is not None else self.rng.random())
        for _ in range(count):
            cx = rng.uniform(spread[0], spread[1]) * self.w
            hh = rng.uniform(*size) * self.h
            r = hh * rng.uniform(0.55, 0.9)
            base = y * self.h
            self.rect(cx - max(2, r * 0.05), base - hh * 0.45, max(3, r * 0.10), hh * 0.45, color, 0.95)
            for k in range(5):
                ox = rng.uniform(-r * 0.55, r * 0.55)
                oy = -hh * 0.55 + rng.uniform(-r * 0.45, r * 0.25)
                self.circle(cx + ox, base + oy, r * rng.uniform(0.45, 0.8), color, 0.92)

    def railing(self, y=0.90, color="#8d939a", gap=0.055, height=0.10, opacity=0.55):
        base = y * self.h
        hh = height * self.h
        x = 0.0
        while x < self.w:
            self.rect(x, base - hh, max(2.0, self.w * 0.0035), hh, color, opacity)
            x += gap * self.w
        self.rect(0, base - hh, self.w, max(2.0, self.h * 0.004), color, opacity)

    # ---- 输出 ----
    def svg(self):
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
                f'viewBox="0 0 {self.w} {self.h}">\n'
                f'<defs>{"".join(self.defs)}</defs>\n'
                + "\n".join(self.body) + "\n</svg>")


# ---------------------------------------------------------------- 场景解释器

def run_scene(scene):
    """把 DSL 场景编译成 SVG 字符串。"""
    c = Compiler(scene.get("w", 1920), scene.get("h", 1080), scene.get("seed", 42))
    for op in scene["ops"]:
        kind = op["type"]
        args = {k: v for k, v in op.items() if k != "type"}
        if kind == "sky":
            c.sky(**args)
        elif kind == "sun":
            c.sun(**args)
        elif kind == "haze":
            c.haze(**args)
        elif kind == "skyline":
            c.skyline(**args)
        elif kind == "ground":
            c.ground(**args)
        elif kind == "grain":
            c.grain(**args)
        elif kind == "vignette":
            c.vignette(**args)
        elif kind == "blur":
            c.blur_layer(**args)
        elif kind == "rect":
            c.rect(**args)
        elif kind == "poly":
            c.poly(**args)
        elif kind == "circle":
            c.circle(**args)
        elif kind == "text":
            c.text(**args)
        elif kind == "landmark":
            c.landmark(**args)
        elif kind == "trees":
            c.trees(**args)
        elif kind == "railing":
            c.railing(**args)
        else:
            raise SystemExit(f"未知图元: {kind}")
    return c.svg()


# ---------------------------------------------------------------- 渲染

def render(svg, out_png, w=1920, h=1080):
    TMP.mkdir(parents=True, exist_ok=True)
    html = TMP / "scene.html"
    html.write_text(
        f'<!doctype html><html><head><meta charset="utf-8">'
        f'<style>html,body{{margin:0;padding:0;background:#000;overflow:hidden}}'
        f'svg{{display:block}}</style></head><body>{svg}</body></html>',
        encoding="utf-8")
    win_html = to_win(html)
    win_out = to_win(Path(out_png).resolve())
    cmd = [CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
           "--force-device-scale-factor=1",
           f"--window-size={w},{h}",
           f"--screenshot={win_out}",
           "--default-background-color=00000000",
           "--virtual-time-budget=2500",
           f"file:///{win_html.replace(chr(92), '/')}"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if not Path(out_png).exists():
        raise SystemExit(f"Chrome 渲染失败: {r.stderr[-400:]}")
    return out_png


def to_win(p):
    p = str(Path(p).resolve())
    if p.startswith("/mnt/") and len(p) > 6:
        return p[5].upper() + ":\\" + p[7:].replace("/", "\\")
    return p


# ---------------------------------------------------------------- demo

DEMO = {
    "w": 1920, "h": 1080, "seed": 20261025,
    "ops": [
        {"type": "sky", "top": "#2b5a8c", "bottom": "#f6d9a8", "horizon": 0.60},
        {"type": "sun", "x": 0.74, "y": 0.40, "r": 0.055, "color": "#fff0c2", "glow": 3.2},
        {"type": "haze", "y": 0.56, "height": 0.22, "color": "#ffe9c9", "opacity": 0.42},
        {"type": "skyline", "horizon": 0.60, "density": 0.8, "depth": 4,
         "base_color": "#33465c", "window_p": 0.30,
         "height_range": [0.10, 0.42], "crowd": 0.85, "seed": 11},
        {"type": "ground", "y": 0.615, "color": "#141a21", "seed": 5},
        {"type": "vignette", "strength": 0.30},
        {"type": "grain", "amount": 0.05, "scale": 1.0, "seed": 7},
    ],
}


def main(argv):
    if len(argv) >= 2 and argv[1] == "--demo":
        out = argv[2] if len(argv) > 2 else str(TMP / "demo.png")
        scene = DEMO
    elif len(argv) >= 3:
        scene = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
        out = argv[2]
    else:
        print(__doc__)
        return 2
    svg = run_scene(scene)
    TMP.mkdir(parents=True, exist_ok=True)
    (TMP / "last.svg").write_text(svg, encoding="utf-8")
    render(svg, out, scene.get("w", 1920), scene.get("h", 1080))
    print(f"OK -> {out}  ({len(svg)} bytes SVG)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
