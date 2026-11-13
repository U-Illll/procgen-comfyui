#!/usr/bin/env python3
"""gen_material_nodes.py — 材质库 → nodes.jsonl（A2 交付）

把 60 套扫描材质的"我能补多少指标缺口"从猜变成量：
  · spectral   : 直接复用 analyze_gap 的 13 指标（同一把尺子）
  · albedo_stats: 读**原图** albedo.jpg（不是已被 gain 归一化到 0.34 的 albedo_256.jpg）
  · scale      : feature_px / min_px / max_px（决定"这个尺寸下该不该用"）
  · pbr/aniso  : roughness 均值、梯度方向主峰

⚠️ 材质必须**等比**测量：analyze_gap.load_rgb() 默认 resize 到 1024×576（非等比），
   方形可平铺纹理会被拉成 16:9 → 人为制造方向性 → orient_entropy / aniso 全被污染。

用法：
    .venv/bin/python gen_material_nodes.py \
        --root ~/go/procgen-lab/materials \
        --out  ~/go/procgen-lab/materials/nodes.jsonl
"""
import argparse
import glob
import hashlib
import json
import math
import os
import sys
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_gap as AG

# 类别 → 可填槽位（按 08 §10.1 的 12 类）
SLOT_BY_CAT = {
    "glass":    ["curtain_wall", "glazing"],
    "metal":    ["mullion", "roof_equip", "railing"],
    "concrete": ["parapet", "podium_stone", "facade_panel"],
    "brick":    ["facade_panel", "podium_stone"],
    "plaster":  ["facade_panel", "parapet"],
    "wood":     ["facade_panel"],
    "stone":    ["podium_stone", "pavement"],
    "asphalt":  ["ground"],
    "pavement": ["pavement", "ground"],
    "tiles":    ["pavement", "roof_equip"],
    "roofing":  ["roof_equip"],
    "fabric":   ["awning"],
}
# 一格的现实尺寸（米）——建筑常识初值，B2 用渲染标定修正
TILE_M_BY_CAT = {
    "glass": (1.2, 1.8), "metal": (0.6, 1.5), "concrete": (1.0, 3.0),
    "brick": (0.6, 1.2), "plaster": (1.0, 3.0), "wood": (0.4, 1.6),
    "stone": (1.0, 2.5), "asphalt": (2.0, 6.0), "pavement": (0.8, 2.0),
    "tiles": (0.4, 1.2), "roofing": (1.0, 3.0), "fabric": (0.5, 1.5),
}
SUITS_KIND = {
    # metal 也要覆盖 skyline：幕墙楼在 skyline 图元里同样有竖梃/屋顶设备
    "glass": ["skyline", "bldg", "landmark"],
    "metal": ["skyline", "bldg", "landmark", "railing", "roof", "roof_equip"],
    "concrete": ["skyline", "bldg", "landmark"], "brick": ["skyline", "bldg"],
    "plaster": ["skyline", "bldg"], "wood": ["skyline", "bldg"],
    "stone": ["ground", "bldg", "landmark"], "asphalt": ["ground"],
    "pavement": ["ground", "railing"], "tiles": ["ground", "roof"],
    "roofing": ["roof"], "fabric": ["awning"],
}
# 无法从贴图可靠反推，用类别先验
METALNESS = {"glass": 0.0, "metal": 0.90, "stone": 0.02, "concrete": 0.0}
FRESNEL = {"glass": 0.05, "metal": 0.65, "fabric": 0.02, "asphalt": 0.03}


# ---------------------------------------------------------------- 测量

def square_copy(path, size=512):
    """等比裁剪 + 缩放到 size×size，落临时 PNG（供 AG.measure 使用）。"""
    im = Image.open(path).convert("RGB")
    side = min(im.size)
    im = im.crop(((im.width - side) // 2, (im.height - side) // 2,
                  (im.width + side) // 2, (im.height + side) // 2))
    im = im.resize((size, size), Image.Resampling.LANCZOS)
    fd, tmp = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    im.save(tmp)
    return tmp


def albedo_stats(path):
    """读原图：mean/std/hue/chroma。数值都在 [0,1]。"""
    a = np.asarray(Image.open(path).convert("RGB").resize((512, 512),
                   Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
    mx, mn = a.max(axis=2), a.min(axis=2)
    chroma = mx - mn
    # 色相：只对高彩度像素取中位数（低彩度像素的 hue 是噪声）
    sel = chroma > 0.06
    hue = float("nan")
    if sel.sum() > 100:
        r, g, b = a[..., 0][sel], a[..., 1][sel], a[..., 2][sel]
        c = chroma[sel]
        h = np.where(mx[sel] == r, ((g - b) / np.maximum(c, 1e-6)) % 6,
             np.where(mx[sel] == g, (b - r) / np.maximum(c, 1e-6) + 2,
                                    (r - g) / np.maximum(c, 1e-6) + 4))
        hue = float(np.median(h * 60.0))
    return {"mean": round(float(a.mean()), 4),
            "std": round(float(a.std()), 4),
            "hue_deg": None if math.isnan(hue) else round(hue, 1),
            "chroma": round(float(chroma.mean()), 4),
            "p05": round(float(np.percentile(a, 5)), 4),
            "p95": round(float(np.percentile(a, 95)), 4)}


def feature_px(g):
    """径向功率谱累计能量 90% 对应的波长（px）。越小=细节越细。"""
    n = min(g.shape)
    win = np.outer(np.hanning(g.shape[0]), np.hanning(g.shape[1]))
    F = np.fft.fftshift(np.abs(np.fft.fft2((g - g.mean()) * win)) ** 2)
    cy, cx = g.shape[0] // 2, g.shape[1] // 2
    y, x = np.mgrid[:g.shape[0], :g.shape[1]]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2).astype(int)
    prof = np.bincount(r.ravel(), F.ravel())[:n // 2]
    if prof.sum() <= 0:
        return None
    prof[0] = 0.0                      # 去掉直流
    cum = np.cumsum(prof) / prof.sum()
    r90 = int(np.searchsorted(cum, 0.90)) + 1
    return round(float(n / max(r90, 1)), 2)


def aniso(g):
    """梯度方向主峰：strength=主峰占比（0.11 各向同性 / >0.3 强方向性），dir_deg=峰角。"""
    gx, gy = AG.grad(g)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    sel = mag > np.percentile(mag, 90)
    if sel.sum() < 200:
        return {"strength": None, "dir_deg": None}
    ang = (np.degrees(np.arctan2(gy[sel], gx[sel])) + 180.0) % 180.0
    hist, edges = np.histogram(ang, bins=36, range=(0, 180))
    p = hist / hist.sum()
    k = int(p.argmax())
    # 与邻窗合并求质心，避免 bin 边界抖动
    idx = [(k - 1) % 36, k, (k + 1) % 36]
    w = np.array([p[i] for i in idx])
    centers = np.array([(edges[i] + edges[i] + 5) / 2 for i in idx])
    strength = float(w.sum())
    # 单峰占比：双峰（正交网格/人字纹）时质心落在两峰之间，dir_deg 没有物理意义
    peak = float(p[k])
    dir_deg = round(float((w * centers).sum() / max(w.sum(), 1e-9)), 1) if peak >= 0.20 else None
    return {"strength": round(strength, 4), "peak_ratio": round(peak, 4), "dir_deg": dir_deg}


def channel_mean(path):
    if not os.path.exists(path):
        return None
    a = np.asarray(Image.open(path).convert("L").resize((256, 256)), dtype=np.float32) / 255.0
    return round(float(a.mean()), 4)


# ---------------------------------------------------------------- 组装

def mat_node(asset_dir, cat):
    mid = os.path.basename(asset_dir)
    alb = os.path.join(asset_dir, "albedo.jpg")
    if not os.path.exists(alb):
        return None
    tmp = square_copy(alb)
    try:
        m = AG.measure(tmp)                      # 13 指标（等比 512×512）
    finally:
        os.unlink(tmp)

    g = np.asarray(Image.open(alb).convert("L").resize((512, 512),
                 Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
    fp = feature_px(g)
    lo, hi = TILE_M_BY_CAT.get(cat, (1.0, 3.0))
    rough = channel_mean(os.path.join(asset_dir, "roughness.jpg"))

    return {
        "mid": mid,
        "cat": cat,
        "paths": {ch: f"{mid}/{ch}_256.jpg" for ch in
                  ("albedo", "normal", "roughness", "ao", "displacement")},
        "res": 256,
        "sha256": hashlib.sha256(open(alb, "rb").read()).hexdigest()[:16],
        "license": "CC0",
        "source": "ambientCG",
        "scale": {
            "measured_res": 512,          # 指标测量尺度（指标与分辨率强相关，必须记录）
            "real_tile_m": round((lo + hi) / 2, 2),
            "recommended_tile_m": [lo, hi],
            "feature_px": fp,
            # 屏幕上一个 tile 小于 ~2 个特征尺度 → 细节全丢；大于 1200px → 重复感明显
            "min_px": None if fp is None else max(12, round(2 * fp, 1)),
            "max_px": 1200,
            "note": "min_px/max_px 初值由 feature_px 推导，B2 用渲染标定修正",
        },
        "albedo_stats": albedo_stats(alb),
        "spectral": {k: m[k] for k in (
            "unique_color_ratio", "grad_midtone_ratio", "freq_slope",
            "orient_entropy", "flat_ratio", "chroma_mean", "hue_entropy")},
        "pbr": {
            "roughness": rough,
            "metalness": METALNESS.get(cat, 0.0),
            "normal_strength": 1.0,
            "fresnel": FRESNEL.get(cat, 0.04),
        },
        "slots": SLOT_BY_CAT.get(cat, []),
        "suits_kind": SUITS_KIND.get(cat, []),
        "valid_under": {"kelvin": [1800, 12000], "ev_min": -2.5,
                        "min_albedo_visible": 0.05},
        "aniso": aniso(g),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("PROCGEN_MATERIALS", str(Path.home() / ".cache" / "procgen" / "materials")))
    ap.add_argument("--out", default=None)
    ap.add_argument("--cat", default=None, help="只处理某一类（调试用）")
    args = ap.parse_args()
    out = args.out or os.path.join(args.root, "nodes.jsonl")

    reg_path = os.path.join(args.root, "registry.json")
    if not os.path.exists(reg_path):
        sys.exit(f"缺 {reg_path}")
    reg = json.load(open(reg_path, encoding="utf-8"))

    # registry: {cat: [assetId, ...]} → 反查，同一 asset 只测一次（跨类去重）
    cat_of = {}
    for cat, ids in reg.items():
        for aid in ids:
            cat_of.setdefault(aid, cat)

    rows, skipped = [], []
    for aid in sorted(cat_of):
        if args.cat and cat_of[aid] != args.cat:
            continue
        d = os.path.join(args.root, aid)
        if not os.path.isdir(d):
            skipped.append((aid, "目录不存在"))
            continue
        n = mat_node(d, cat_of[aid])
        if n is None:
            skipped.append((aid, "缺 albedo.jpg"))
            continue
        rows.append(n)
        print(f"  {n['cat']:<9} {aid:<16} feature_px={n['scale']['feature_px']:<6} "
              f"orient={n['spectral']['orient_entropy']:<6} "
              f"aniso={n['aniso']['strength']}@{n['aniso']['dir_deg']}")

    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n→ {out}：{len(rows)} 条" + (f"，跳过 {len(skipped)}: {skipped}" if skipped else ""))

    # 抽检：同类方差 + 跨类方向性（写入 stderr 便于观察，不污染 jsonl）
    if rows:
        cats = sorted({r["cat"] for r in rows})
        print("\n跨类抽检（orient_entropy 越高越各向同性）：")
        for c in cats:
            vals = [r["spectral"]["orient_entropy"] for r in rows if r["cat"] == c]
            print(f"  {c:<9} n={len(vals)} orient_mean={np.mean(vals):.3f} "
                  f"sd={np.std(vals):.4f}")
        br = [r for r in rows if r["cat"] == "brick"]
        if br:
            print("\nbrick 方向性（dir_deg=90 为横纹；None=双峰/无主导方向）：",
                  [(r["mid"], r["aniso"]["dir_deg"], r["aniso"]["peak_ratio"]) for r in br])


if __name__ == "__main__":
    main()
