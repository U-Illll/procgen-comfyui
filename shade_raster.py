#!/usr/bin/env python3
"""shade_raster.py — P1-1/P1-2：逐像素 UV + 法线光照（numpy 域）

输入（pipeline 的 build/zbuf 阶段）：
  --zdir   zbuf 产物：ids.npy / zbuf.npy / zmeta.json（id_names/id_boxes/id_kinds）
  --meta   build/scene_meta.json：light / materials / nodes(depth)
  --base   build/scene_beauty.png（SVG 底图：背景 + 语义细节）
  --materials 材质根目录
输出：
  --out    beauty（前景材质细节 + 背景保留）

两种模式：
  overlay（默认）：材质作为**乘性微观起伏**，保留 SVG 的窗格/灯带/结构；
                   detail 在 id 接缝处做 1px 羽化，避免硬边抬高梯度峰度。
  replace：整面重绘（albedo × N·L + GGX），用于纯材质面/实验。

用法：
  .venv/bin/python shade_raster.py --zdir DIR --meta build/scene_meta.json \
      --base build/scene_beauty.png --materials ~/go/procgen-lab/materials --out r.png
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

HOME = Path.home()
PROCGEN = HOME / "go" / "procgen"
sys.path.insert(0, str(PROCGEN))
import zbuf as zb          # normalized_depth
import procgen2 as pg      # kelvin_to_rgb


def load_rgb(p):
    return np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0


def load_tex(root, asset, name, blur=0.0):
    for fn in (f"{name}_1024.jpg", f"{name}.jpg", f"{name}_512.jpg", f"{name}_256.jpg"):
        p = Path(root) / asset / fn
        if p.exists():
            a = load_rgb(p)
            if blur > 0:
                im = Image.fromarray((a * 255).astype(np.uint8)).filter(
                    ImageFilter.GaussianBlur(blur))
                a = np.asarray(im, np.float32) / 255.0
            return a
    return None


def bilinear(tex, u, v):
    """u/v 是 tile 坐标（可 >1），返回 (N,3)。"""
    h, w = tex.shape[:2]
    x = (u % 1.0) * w
    y = (v % 1.0) * h
    xf = np.floor(x)
    yf = np.floor(y)
    x0 = xf.astype(np.int32) % w
    y0 = yf.astype(np.int32) % h
    x1 = (x0 + 1) % w
    y1 = (y0 + 1) % h
    fx = (x - xf)[:, None]
    fy = (y - yf)[:, None]
    return (tex[y0, x0] * (1 - fx) * (1 - fy) + tex[y0, x1] * fx * (1 - fy)
            + tex[y1, x0] * (1 - fx) * fy + tex[y1, x1] * fx * fy)


def hex_rgb(s):
    s = s.lstrip("#")
    return np.array([int(s[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zdir", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--materials", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=("overlay", "replace"), default="overlay")
    ap.add_argument("--detail-strength", type=float, default=0.25,
                    help="overlay 的材质起伏强度（0=无，1=原始对比）")
    ap.add_argument("--tex-blur", type=float, default=1.2,
                    help="albedo 采样前的高斯模糊（模拟镜头 MTF；0=关）")
    ap.add_argument("--normal-strength", type=float, default=0.65)
    ap.add_argument("--spec", type=float, default=0.55)
    ap.add_argument("--seam-blur", type=float, default=1.0,
                    help="overlay detail 的接缝羽化半径（px）")
    args = ap.parse_args()

    zdir = Path(args.zdir)
    ids = np.load(zdir / "ids.npy")
    z = np.load(zdir / "zbuf.npy")
    zm = json.loads((zdir / "zmeta.json").read_text(encoding="utf-8"))
    meta = json.loads(Path(args.meta).read_text(encoding="utf-8"))
    light = meta.get("light") or {}
    mats = meta.get("materials") or {}
    depths = {n["id"]: n.get("depth", 0.5) for n in meta.get("nodes", [])}

    az = math.radians(light.get("azimuth", 250.0))
    el = math.radians(max(1.0, light.get("elevation", 12.0)))
    L = np.array([math.sin(az) * math.cos(el), math.sin(el),
                  -math.cos(az) * math.cos(el)], np.float32)
    L /= np.linalg.norm(L) + 1e-6
    V = np.array([0.0, 0.0, 1.0], np.float32)
    Hv = L + V
    Hv /= np.linalg.norm(Hv) + 1e-6
    intensity = float(light.get("intensity", 1.0))
    ambient = float(light.get("ambient", 0.28))
    haze = float(light.get("haze", 0.34))
    hzc = hex_rgb(light.get("haze_color", "#cfe0ee"))
    lcol = hex_rgb(pg.kelvin_to_rgb(float(light.get("kelvin", 3200.0))))

    base = load_rgb(args.base)
    h, w = ids.shape
    if base.shape[:2] != (h, w):
        base = np.asarray(Image.open(args.base).convert("RGB")
                          .resize((w, h), Image.Resampling.LANCZOS), np.float32) / 255.0
    dn = zb.normalized_depth(z)
    out = base.copy()

    # overlay 的累积图（循环后统一羽化，避免逐 id 硬边）
    dmap = np.ones((h, w), np.float32)
    smap = np.zeros((h, w, 3), np.float32)
    n_done = 0

    for sid, name in (zm.get("id_names") or {}).items():
        i = int(sid)
        asset = mats.get(name)
        if not asset:
            continue
        alb = load_tex(args.materials, asset, "albedo", args.tex_blur)
        if alb is None:
            continue
        m = ids == i
        if not m.any():
            continue
        ys, xs = np.nonzero(m)
        box = (zm.get("id_boxes") or {}).get(sid) or [0, 0, w, h]
        x0, y0, x1, y1 = [float(v) for v in box]
        depth = float(depths.get(name, 0.5))
        tile_px = 256.0 * 0.30 * (0.70 + 0.60 * depth)
        ut = xs / tile_px
        vt = ys / tile_px

        A = bilinear(alb, ut, vt)
        nrm = load_tex(args.materials, asset, "normal")
        rgh = load_tex(args.materials, asset, "roughness")
        kind = (zm.get("id_kinds") or {}).get(sid, "bldg")

        if nrm is not None:
            N = bilinear(nrm, ut, vt) * 2.0 - 1.0
            Nn = (np.stack([N[:, 0], -N[:, 2], N[:, 1]], 1) if kind == "ground"
                  else N.copy())
            Nn[:, :2] *= args.normal_strength
            Nn /= np.linalg.norm(Nn, axis=1, keepdims=True) + 1e-6
            ny = Nn[:, 1]
            ndl = np.maximum(0.0, Nn @ L)
            ndh = np.maximum(0.0, Nn @ Hv)
        else:
            ny = 0.0
            ndl = np.full(xs.shape, max(0.0, float(L[2])), np.float32)
            ndh = np.full(xs.shape, max(0.0, float(Hv[2])), np.float32)

        k = ambient + intensity * ndl + 0.12 * np.maximum(0.0, ny)
        r = (bilinear(rgh, ut, vt).mean(axis=1) if rgh is not None
             else np.full(xs.shape, 0.7, np.float32))
        shin = 4.0 + (1.0 - r) * 60.0
        spec = (1.0 - r) ** 2 * np.power(ndh, shin) * args.spec
        fog = np.power(1.0 - dn[m], 1.15)[:, None] * haze

        if args.mode == "overlay":
            dmap[ys, xs] = 1.0 + (A.mean(axis=1) / 0.34 - 1.0) * args.detail_strength
            smap[ys, xs] = lcol[None, :] * (spec[:, None] * 0.5)
        else:
            col = A * k[:, None] + lcol[None, :] * spec[:, None]
            col = col * (1.0 - fog) + hzc[None, :] * fog
            out[ys, xs] = np.clip(col, 0.0, 1.0)
        n_done += 1

    if args.mode == "overlay":
        if args.seam_blur > 0:
            dm8 = np.clip(dmap / 2.0, 0, 1) * 255
            dm8 = Image.fromarray(dm8.astype(np.uint8)).filter(
                ImageFilter.GaussianBlur(args.seam_blur))
            dmap = np.asarray(dm8, np.float32) / 255.0 * 2.0
        out = np.clip(base * dmap[..., None] + smap, 0.0, 1.0)

    # 前景边界 1px 羽化，避免与 SVG 底图硬切
    fg = (ids >= 0).astype(np.uint8) * 255
    fg = np.asarray(Image.fromarray(fg).filter(ImageFilter.GaussianBlur(1.0)),
                    np.float32)[..., None] / 255.0
    res = out * fg + base * (1.0 - fg)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(res, 0, 1) * 255 + 0.5).astype(np.uint8)).save(args.out)
    print(f"raster: {n_done} 图元 → {args.out}  mode={args.mode}")


if __name__ == "__main__":
    main()
