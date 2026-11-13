#!/usr/bin/env python3
"""fetch_refs.py — 从 picsum（Unsplash 精选，Unsplash License）挑「城市/建筑」摄影做参考集

为什么不能直接搜图库：Unsplash / Pexels / Pixabay 的搜索页在本地都 403 或 404（反爬），
Wikimedia 超时。picsum 是 Unsplash 的稳定镜像列表（993 张，含 id/author/原始 unsplash 页），
CDN 直连可达，许可同为 Unsplash License。

挑选逻辑（无人值守，靠低层特征打分）：
  · 垂直边缘密度高  → 建筑立面（最强信号）
  · 上 1/3 平滑度   → 有天空（城市/风景构图）
  · 细节熵（灰度 std）→ 排除纯色/极简/糊图
  · 水平边缘惩罚     → 排除纯海平面/地平线/条纹

用法：
    python fetch_refs.py --top 12            # 打分取前 12 张，下 2560×1600
    python fetch_refs.py --top 12 --scan 300 # 只扫前 300 张缩略（更快）
"""
import argparse
import concurrent.futures as cf
import glob
import json
import os
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

UA = "procgen-lab/0.1 (local research)"
OUT = os.environ.get("PROCGEN_REFERENCE_OUT", str(Path.home() / "Pictures" / "procgen-reference"))
THUMB = (420, 280)


def get(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def load_list():
    items = []
    for p in sorted(glob.glob("/tmp/picsum_p*.json")):
        try:
            items += json.load(open(p, encoding="utf-8"))
        except Exception:
            pass
    return items


def _erode(m):
    return (m[1:-1, 1:-1] & m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:])


def long_line_ratio(g, thr=0.07, iters=4):
    """长直线密度：连续腐蚀后存活的边缘像素。

    建筑立面 = 长直边（窗框/幕墙/楼体轮廓）；树林/海浪/草地 = 短碎边。
    腐蚀 4 次后只剩长度 > ~9px 的边，这比「边缘多就是建筑」可靠得多。
    """
    gx = np.zeros_like(g); gy = np.zeros_like(g)
    gx[:, 1:-1] = (g[:, 2:] - g[:, :-2]) * 0.5
    gy[1:-1, :] = (g[2:, :] - g[:-2, :]) * 0.5
    mv = np.abs(gx) > thr
    mh = np.abs(gy) > thr
    for _ in range(iters):
        if min(mv.shape) < 6:
            break
        mv = _erode(mv)
        mh = _erode(mh)
    return float(mv.mean()) + 0.45 * float(mh.mean())


def green_ratio(a):
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    return float(np.mean((g > r * 1.12) & (g > b * 1.12) & (g > 0.14)))


def score(im):
    a = np.asarray(im.convert("RGB"), np.float32) / 255.0
    g = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    lines = long_line_ratio(g)
    green = green_ratio(a)
    top = g[: g.shape[0] // 3]
    sky = 1.0 - float(np.std(top))          # 上 1/3 平滑 = 有天空
    detail = float(np.std(g))
    return 40.0 * lines + 1.2 * sky + 0.8 * detail - 3.0 * green


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--scan", type=int, default=400)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    items = load_list()[: args.scan]
    print(f"候选 {len(items)} 张，取缩略打分…")

    def probe(it):
        try:
            data = get(f"https://picsum.photos/id/{it['id']}/{THUMB[0]}/{THUMB[1]}")
            import io
            im = Image.open(io.BytesIO(data))
            return it, score(im)
        except Exception:
            return it, -1.0

    scored = []
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        for i, (it, s) in enumerate(ex.map(probe, items)):
            if s > 0:
                scored.append((s, it))
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(items)}")

    scored.sort(key=lambda x: -x[0])
    picked = scored[: args.top]
    print(f"\n入选 {len(picked)} 张：")
    for s, it in picked:
        print(f"  {s:.3f}  id={it['id']:<6} {it['width']}x{it['height']}  "
              f"{it['author'][:22]:<22} {it['url']}")

    ok = 0
    for s, it in picked:
        dst = os.path.join(args.out, f"ref-{it['id']}-{it['width']}x{it['height']}.jpg")
        if os.path.exists(dst):
            ok += 1
            continue
        try:
            data = get(f"https://picsum.photos/id/{it['id']}/2560/1600", timeout=90)
            with open(dst, "wb") as f:
                f.write(data)
            ok += 1
        except Exception as e:
            print(f"  ! {it['id']}: {e}")
    print(f"\n下载完成 {ok}/{len(picked)} -> {args.out}")
    with open(os.path.join(args.out, "SOURCES.json"), "w", encoding="utf-8") as f:
        json.dump([{"id": it["id"], "author": it["author"], "unsplash": it["url"],
                    "license": "Unsplash License", "score": round(s, 4)}
                   for s, it in picked], f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
