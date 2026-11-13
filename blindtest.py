#!/usr/bin/env python3
"""blindtest.py — 独立盲测：合成图 vs 真实照片，纹理级可分性（审核轮 1 要求）

为什么必须做（审核轮 1 的最大 P0）：
    调参与验收用的是同一套 13 指标 + 同一组 5 张参考图 → 典型的 Goodhart。
    `rd_r1` 下降不能证明"更像照片"，只能证明"更像那 5 张的统计量"。
    这里做一个**独立**判别：不看整图，只看 96×96 的局部 patch，
    特征刻意**避开 13 指标**里的任何一项，且真实集用**未参与调参**的 ref-pro 24 张。

判据：
    AUC ≈ 0.5 → 纹理级不可分（强证据，支持"已接近照片"）
    AUC > 0.8 → 仍明显可分（`rd_r1` 的改善与观感脱节）

用法：
    .venv/bin/python blindtest.py \
      --syn "/mnt/c/.../procgen/c2out/*_c2.png" \
      --real "/mnt/c/.../assets/ref-pro/*.jpg" \
      --patch 96 --per-syn 30 --per-real 8 --out /tmp/blind
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

RNG = np.random.RandomState(20261027)


# ---------------------------------------------------------------- 特征
# 刻意不用 13 指标里的：noise_sigma/blockiness/freq_slope/grad_kurtosis/step_ratio/
# grad_midtone/periodicity/unique_color/chroma/hue_entropy/flat/local_contrast_cv/orient_entropy
# 只用"局部结构 + 色彩结构"这一族，避免与验收指标同源。

def lbp_entropy(g, bins=16):
    """4 邻域 LBP（16 码）的熵：纹理的局部符号结构。"""
    c = g[1:-1, 1:-1]
    code = ((g[:-2, 1:-1] > c).astype(np.uint8)
            | ((g[2:, 1:-1] > c).astype(np.uint8) << 1)
            | ((g[1:-1, :-2] > c).astype(np.uint8) << 2)
            | ((g[1:-1, 2:] > c).astype(np.uint8) << 3))
    h = np.bincount(code.ravel(), minlength=bins).astype(np.float64)
    p = h / max(h.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum() / np.log2(bins))


def block_var_cv(g, b=8):
    """8×8 块内方差的变异系数：表面起伏的"不均匀程度"。"""
    h, w = g.shape
    hh, ww = h // b * b, w // b * b
    v = g[:hh, :ww].reshape(hh // b, b, ww // b, b).var(axis=(1, 3))
    return float(v.std() / max(v.mean(), 1e-9))


def hf_ratio(g):
    """高频能量占比（>1/4 Nyquist）：细节密度。"""
    f = np.abs(np.fft.rfft2(g - g.mean())) ** 2
    fy = f.shape[0]
    y, x = np.mgrid[:fy, :f.shape[1]]
    r = np.sqrt((y / fy) ** 2 + (x / f.shape[1]) ** 2)
    tot = f.sum()
    return float(f[r > 0.25].sum() / max(tot, 1e-9))


def edge_width(g):
    """边缘半高宽的近似：强梯度处的二阶导/一阶导比。越宽 = 光学越软。"""
    gx = np.zeros_like(g); gy = np.zeros_like(g)
    gx[:, 1:-1] = (g[:, 2:] - g[:, :-2]) * 0.5
    gy[1:-1, :] = (g[2:, :] - g[:-2, :]) * 0.5
    mag = np.sqrt(gx * gx + gy * gy)
    strong = mag > np.percentile(mag, 90)
    if strong.sum() < 50:
        return 0.0
    lap = np.zeros_like(g)
    lap[:, 1:-1] = g[:, 2:] - 2 * g[:, 1:-1] + g[:, :-2]
    return float(np.median(np.abs(lap[strong])) / max(np.median(mag[strong]), 1e-9))


def color_structure(a):
    """跨通道差分结构（不是彩度均值，避免与 chroma 同源）。"""
    r, gg, b = a[..., 0], a[..., 1], a[..., 2]
    rg, bg = r - gg, b - gg
    # 通道差的空间梯度能量比：光照一致性的代理
    g_rg = np.abs(np.diff(rg, axis=1)).mean() + np.abs(np.diff(rg, axis=0)).mean()
    g_bg = np.abs(np.diff(bg, axis=1)).mean() + np.abs(np.diff(bg, axis=0)).mean()
    return float((g_rg + g_bg) / max(np.abs(rg).mean() + np.abs(bg).mean(), 1e-9))


FEATURE_NAMES = ["lbp_entropy", "block_var_cv", "hf_ratio",
                 "edge_width", "color_struct"]


def features(patch_rgb):
    g = (0.2126 * patch_rgb[..., 0] + 0.7152 * patch_rgb[..., 1]
         + 0.0722 * patch_rgb[..., 2]).astype(np.float32)
    return np.array([lbp_entropy(g), block_var_cv(g), hf_ratio(g),
                     edge_width(g), color_structure(patch_rgb)], dtype=np.float64)


def patch_feats(path, psize=96, per=20, target_w=None):
    im = Image.open(path).convert("RGB")
    # 视场对齐：真实照片 2560px、合成图 1920px，直接取 96px patch 时
    # 覆盖的物理视场不同（尺度混杂会抬高 AUC）。统一缩到同一宽度后再采样。
    if target_w and im.width != target_w:
        im = im.resize((target_w, max(1, round(im.height * target_w / im.width))),
                       Image.Resampling.LANCZOS)
    a = np.asarray(im, dtype=np.float32) / 255.0
    h, w = a.shape[:2]
    if h < psize or w < psize:
        return []
    out = []
    for _ in range(per):
        y = RNG.randint(0, h - psize)
        x = RNG.randint(0, w - psize)
        out.append(features(a[y:y + psize, x:x + psize]))
    return out


# ---------------------------------------------------------------- 分类器

def logistic_fit(X, y, lr=0.15, epochs=4000, l2=1e-3):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(epochs):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        grad_w = X.T @ (p - y) / n + l2 * w
        grad_b = float(np.mean(p - y))
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def auc(y, score):
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def cv_auc(X, y, folds=5):
    idx = RNG.permutation(len(y))
    aucs = []
    for k in range(folds):
        te = idx[k::folds]
        tr = np.setdiff1d(idx, te)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        w, b = logistic_fit((X[tr] - mu) / sd, y[tr])
        s = ((X[te] - mu) / sd) @ w + b
        aucs.append(auc(y[te], s))
    return float(np.nanmean(aucs)), aucs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--syn", nargs="+", required=True)
    ap.add_argument("--real", nargs="+", required=True)
    ap.add_argument("--patch", type=int, default=96)
    ap.add_argument("--per-syn", type=int, default=30)
    ap.add_argument("--per-real", type=int, default=8)
    ap.add_argument("--target-width", type=int, default=1536,
                    help="两边统一缩到该宽度再取 patch（视场对齐；0=不缩放）")
    ap.add_argument("--out", default="/tmp/blind")
    args = ap.parse_args()

    S = [p for pat in args.syn for p in sorted(glob.glob(pat))]
    R = [p for pat in args.real for p in sorted(glob.glob(pat))]
    if not S or not R:
        sys.exit(f"输入为空 syn={S} real={R}")

    Xs = [f for p in S for f in patch_feats(p, args.patch, args.per_syn, args.target_width)]
    Xr = [f for p in R for f in patch_feats(p, args.patch, args.per_real, args.target_width)]
    X = np.vstack(Xs + Xr)
    y = np.r_[np.ones(len(Xs)), np.zeros(len(Xr))]

    mu, sd = X.mean(0), X.std(0) + 1e-9
    w, b = logistic_fit((X - mu) / sd, y)
    s = ((X - mu) / sd) @ w + b
    auc_all = auc(y, s)
    cv, folds = cv_auc(X, y)

    os.makedirs(args.out, exist_ok=True)
    res = {
        "syn_files": len(S), "real_files": len(R),
        "syn_patches": len(Xs), "real_patches": len(Xr),
        "feature_names": FEATURE_NAMES,
        "feature_weight": {n: round(float(v), 4) for n, v in zip(FEATURE_NAMES, w)},
        "auc_in_sample": round(auc_all, 4),
        "auc_cv5": round(cv, 4),
        "auc_folds": [round(float(a), 4) for a in folds],
        "verdict": ("纹理级不可分（AUC≤0.6）" if cv <= 0.6 else
                    "部分可分（0.6<AUC≤0.8）" if cv <= 0.8 else
                    "明显可分（AUC>0.8）：rd_r1 的改善与观感脱节"),
        "note": ("真实集必须用未参与调参的照片（ref-pro）；特征已避开 13 指标族。"
                 "判别器为手写逻辑回归 + 5 折 CV，无第三方依赖。"),
    }
    with open(os.path.join(args.out, "blindtest.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    print(f"合成 {len(S)} 图 / {len(Xs)} patch｜真实 {len(R)} 图 / {len(Xr)} patch")
    print("特征权重：" + ", ".join(f"{n}={v:+.2f}" for n, v in zip(FEATURE_NAMES, w)))
    print(f"AUC（全量）= {auc_all:.3f}｜AUC（5 折 CV）= {cv:.3f}")
    print(f"结论：{res['verdict']}")
    print(f"→ {os.path.join(args.out, 'blindtest.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
