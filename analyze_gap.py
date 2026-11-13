#!/usr/bin/env python3
"""analyze_gap.py — 合成图 vs 真实照片的「低层成像统计」差距归因

目的：不靠主观，把「一眼假」量化成可复现的数字，并按「修复成本」排序。

用法：
    python analyze_gap.py --syn assets/*.png --real assets/ref/*.jpg --out report/

指标分三族：
  A 成像链特征（题材弱相关，可直接比）：noise / blockiness / freq_slope / grad_kurtosis
  B 材质与色彩特征：unique_color_ratio / chroma_std / hue_entropy / sat_luma_corr
  C 结构与细节分布：flat_ratio / edge_frac / local_contrast_cv / orient_entropy
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
RESAMPLE = Image.Resampling.LANCZOS


# ---------------------------------------------------------------- 基础工具
def load_rgb(path, size=(1024, 576)):
    im = Image.open(path).convert("RGB")
    if size is not None:
        im = im.resize(size, RESAMPLE)
    return np.asarray(im, dtype=np.float32) / 255.0


def to_luma(a):
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def grad(g):
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:-1] = (g[:, 2:] - g[:, :-2]) * 0.5
    gy[1:-1, :] = (g[2:, :] - g[:-2, :]) * 0.5
    return gx, gy


# ---------------------------------------------------------------- A 成像链
def noise_sigma(g):
    """Immerkær 1996 快速噪声估计：对 3x3 拉普拉斯型核响应取均值。"""
    r = (
        4 * g[1:-1, 1:-1]
        - g[:-2, 1:-1] - g[2:, 1:-1]
        - g[1:-1, :-2] - g[1:-1, 2:]
    )
    return float(np.sqrt(np.pi / 2) * np.mean(np.abs(r)) / 6.0)


def blockiness(g, block=8):
    """JPEG 块效应：块边界处的水平/垂直差分 相对 块内差分 的比值。
    纯合成 PNG 应接近 1.0（无 8px 周期性），JPEG 压缩图 > 1.05。"""
    gx, gy = grad(g)
    dx = np.abs(gx)
    dy = np.abs(gy)
    h, w = dx.shape
    cols = np.arange(block, w - 1, block)
    rows = np.arange(block, h - 1, block)
    on = np.concatenate([dx[:, cols].ravel(), dy[rows, :].ravel()])
    inner_cols = np.array([c for c in range(1, w - 1) if c % block])
    inner_rows = np.array([r for r in range(1, h - 1) if r % block])
    off = np.concatenate([dx[:, inner_cols].ravel(), dy[inner_rows, :].ravel()])
    # 用中位数而非均值：等距的建筑竖边会落在 8 的倍数上，均值法会被它带偏
    on_m, off_m = float(np.median(on)), float(np.median(off))
    # 纯色/近纯色图上「块内差分」≈0，比值会放大成 1e5 级噪声（实测 Metal049A=586272）。
    # 这类图没有 8px 周期性可言 → 返回 None，由调用方剔除，不要伪造一个巨大的 blockiness。
    if off_m < 1e-6:
        return None
    return on_m / off_m


def grad_midtone_ratio(g, floor=1e-4):
    """梯度幅值的「中间调」占比（CG 感的核心量化）。
    归一化到 p99.5 后统计 0.15~0.6 区间：
      · 真实照片：光学 MTF + 材质微观起伏 → 连续谱，中间调多
      · 矢量/参数化：要么平坦纯色，要么 1px 强台阶 → 中间调塌陷（双峰）
    """
    gx, gy = grad(g)
    m = np.sqrt(gx ** 2 + gy ** 2)
    top = np.percentile(m, 99.5)
    if top <= floor:
        return 0.0
    mn = m / top
    mid = (mn > 0.15) & (mn < 0.60)
    tails = (mn <= 0.15) | (mn >= 0.60)
    return float(mid.sum() / max(1, tails.sum()))


def radial_spectrum_slope(g):
    """径向平均功率谱 log-log 斜率。自然图像 ≈ -2（1/f²）；矢量插画高频缺失，更陡。"""
    win = np.outer(np.hanning(g.shape[0]), np.hanning(g.shape[1]))
    F = np.fft.fftshift(np.abs(np.fft.fft2((g - g.mean()) * win)) ** 2)
    h, w = g.shape
    cy, cx = h // 2, w // 2
    y, x = np.mgrid[:h, :w]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2).astype(int)
    rmax = min(cy, cx) - 1
    prof = np.bincount(r.ravel(), F.ravel())[:rmax]
    cnt = np.bincount(r.ravel())[:rmax]
    prof = prof / np.maximum(cnt, 1)
    k = np.arange(1, rmax)
    ok = prof[k] > 0
    if ok.sum() < 10:
        return 0.0
    lx = np.log(k[ok])
    ly = np.log(prof[k][ok])
    return float(np.polyfit(lx, ly, 1)[0])


def periodicity(g):
    """结构周期性：FFT 幅度谱里的孤立强峰。

    等距栏杆 / 均布窗格 / 等宽楼距 会在频谱上留下**离散峰**——这是「重复元素」
    的物理指纹。真实照片的窗户也周期性，但伴随大量不规则细节把峰盖住了。
    返回 log10(峰值/中位数)；> 2.0 说明图里存在「一眼看出是阵列」的结构。
    """
    win = np.outer(np.hanning(g.shape[0]), np.hanning(g.shape[1]))
    F = np.abs(np.fft.fftshift(np.fft.fft2((g - g.mean()) * win)))
    h, w = g.shape
    cy, cx = h // 2, w // 2
    y, x = np.mgrid[:h, :w]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    F = F * (r > 12)                 # 去低频（整体明暗、画面边框）
    # 注意：不能清零横/纵轴线！等距竖条(Lattice)→水平轴上的峰，等距横线→垂直轴上的峰，
    # 轴线恰恰是「重复元素」信号所在，清了就等于把要测的东西删了。
    q = F[cy:, cx:]                  # 只取一个象限（其余是共轭对称）
    if q.size == 0:
        return 0.0
    return float(np.log10(q.max() / (np.median(q) + 1e-9) + 1e-9))


def grad_kurtosis(g):
    """梯度幅值峰度：矢量的抗锯齿边缘是强台阶 → 长尾；光学成像边缘被 MTF 展宽 → 尾部短。"""
    gx, gy = grad(g)
    m = np.sqrt(gx ** 2 + gy ** 2).ravel()
    m = m - m.mean()
    s = m.std() + 1e-12
    return float(np.mean((m / s) ** 4))


def step_ratio(g, thr=0.16):
    """1px 强台阶比例：相邻像素差 > thr 的像素占比（矢量边缘的指纹）。"""
    gx, gy = grad(g)
    m = np.sqrt(gx ** 2 + gy ** 2)
    return float(np.mean(m > thr))


# ---------------------------------------------------------------- B 材质/色彩
def unique_color_ratio(a):
    q = np.clip(a * 255.0 + 0.5, 0, 255).astype(np.uint8)
    flat = q.reshape(-1, 3)
    packed = (flat[:, 0].astype(np.uint32) << 16) | (flat[:, 1].astype(np.uint32) << 8) | flat[:, 2]
    return float(len(np.unique(packed)) / max(1, flat.shape[0]))


def chroma_stats(a):
    mx = a.max(2)
    mn = a.min(2)
    chroma = (mx - mn) / np.maximum(mx, 1e-6)
    # 简易 Lab 的 a/b 代理：RG 差 与 YB 差
    rg = a[..., 0] - a[..., 1]
    yb = 0.5 * (a[..., 0] + a[..., 1]) - a[..., 2]
    return float(chroma.mean()), float(rg.std()), float(yb.std())


def hue_entropy(a, bins=48):
    mx = a.max(2)
    mn = a.min(2)
    d = mx - mn
    sat = np.where(mx > 1e-6, d / np.maximum(mx, 1e-6), 0)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    hue = np.zeros_like(mx)
    m = d > 1e-6
    idx = (mx == r) & m
    hue[idx] = ((g - b)[idx] / d[idx]) % 6
    idx = (mx == g) & m
    hue[idx] = ((b - r)[idx] / d[idx]) + 2
    idx = (mx == b) & m
    hue[idx] = ((r - g)[idx] / d[idx]) + 4
    hue = hue / 6.0
    sel = sat > 0.08
    if sel.sum() < 100:
        return 0.0
    hist, _ = np.histogram(hue[sel], bins=bins, range=(0, 1))
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def sat_luma_corr(a):
    mx = a.max(2)
    mn = a.min(2)
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0)
    y = to_luma(a)
    s, yy = sat.ravel(), y.ravel()
    if s.std() < 1e-6:
        return 0.0
    return float(np.corrcoef(s, yy)[0, 1])


# ---------------------------------------------------------------- C 结构/细节
def flat_ratio(g, thr=0.006):
    gx, gy = grad(g)
    return float(np.mean(np.sqrt(gx ** 2 + gy ** 2) < thr))


def local_contrast_cv(g, win=16):
    h, w = g.shape
    hh, ww = h // win * win, w // win * win
    b = g[:hh, :ww].reshape(hh // win, win, ww // win, win).transpose(0, 2, 1, 3)
    std = b.reshape(hh // win, ww // win, -1).std(-1)
    return float(std.std() / (std.mean() + 1e-9))


def orient_entropy(g, bins=18):
    gx, gy = grad(g)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    sel = mag > np.percentile(mag, 90)
    if sel.sum() < 200:
        return 0.0
    ang = (np.arctan2(gy[sel], gx[sel]) + np.pi) % np.pi
    hist, _ = np.histogram(ang, bins=bins, range=(0, np.pi))
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum() / np.log2(bins))


# ---------------------------------------------------------------- 汇总
def measure(path):
    a = load_rgb(path)
    g = to_luma(a)
    # 块效应必须在原尺寸上测：resize 会抹掉 8px 周期性
    gfull = to_luma(load_rgb(path, size=None))
    cm, rg_s, yb_s = chroma_stats(a)
    blk = blockiness(gfull)
    return {
        "file": os.path.basename(path),
        # A 成像链
        "noise_sigma": round(noise_sigma(g), 5),
        # 纯色图无 8px 周期性可言 → blockiness 为 None（不是 0，也不是天文数字）
        "blockiness": round(blk, 4) if blk is not None else None,
        "freq_slope": round(radial_spectrum_slope(g), 3),
        "grad_kurtosis": round(grad_kurtosis(g), 2),
        "step_ratio": round(step_ratio(g), 5),
        "grad_midtone_ratio": round(grad_midtone_ratio(g), 4),
        "periodicity": round(periodicity(g), 3),
        # B 色彩/材质
        "unique_color_ratio": round(unique_color_ratio(a), 5),
        "chroma_mean": round(cm, 4),
        "rg_std": round(rg_s, 4),
        "yb_std": round(yb_s, 4),
        "hue_entropy": round(hue_entropy(a), 3),
        # C 结构
        "flat_ratio": round(flat_ratio(g), 4),
        "local_contrast_cv": round(local_contrast_cv(g), 3),
        "orient_entropy": round(orient_entropy(g), 4),
    }


KEYS = [
    ("noise_sigma", "噪声 σ（成像链）", "真实有传感器噪声；矢量≈0"),
    ("blockiness", "JPEG 块效应比", ">1.05 有 8px 周期；纯 PNG ≈1.0"),
    ("freq_slope", "功率谱斜率", "自然≈-2；高频缺失则更负"),
    ("grad_kurtosis", "梯度峰度", "矢量台阶→长尾（大）；光学展宽→小"),
    ("step_ratio", "1px 强台阶占比", "矢量边缘指纹"),
    ("grad_midtone_ratio", "梯度中间调/两端", "低=要么纯色要么硬边（CG 感）"),
    ("periodicity", "结构周期峰 log10", "高=存在等距阵列（栏杆/窗格/楼距）"),
    ("unique_color_ratio", "唯一色占比", "矢量色块化→极低"),
    ("chroma_mean", "彩度均值", "真实材质有微观色差"),
    ("hue_entropy", "色相熵", "低=配色单调"),
    ("flat_ratio", "平坦区占比", "高=大面积纯色块"),
    ("local_contrast_cv", "细节密度不均匀度", "高=大片空 + 强边（双峰，假）"),
    ("orient_entropy", "梯度方向熵(归一)", "低=只有水平/垂直（假）"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--syn", nargs="+", required=True)
    ap.add_argument("--real", nargs="+", required=True)
    ap.add_argument("--out", default=".")
    args = ap.parse_args()

    syn = [p for pat in args.syn for p in glob.glob(pat)]
    real = [p for pat in args.real for p in glob.glob(pat)]
    if not syn or not real:
        sys.exit(f"输入为空: syn={len(syn)} real={len(real)}")

    S = [measure(p) for p in syn]
    R = [measure(p) for p in real]

    def avg(rows, k):
        return float(np.mean([r[k] for r in rows]))

    os.makedirs(args.out, exist_ok=True)
    report = {"synthetic": S, "real": R, "comparison": []}
    lines = []
    for k, cn, why in KEYS:
        s, r = avg(S, k), avg(R, k)
        ratio = (s / r) if abs(r) > 1e-9 else float("nan")
        report["comparison"].append(
            {"key": k, "name": cn, "synthetic": round(s, 5), "real": round(r, 5),
             "ratio": round(ratio, 3), "why": why}
        )
        lines.append((k, cn, s, r, ratio, why))

    with open(os.path.join(args.out, "gap_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n合成 {len(S)} 张 / 真实 {len(R)} 张\n")
    print(f"{'指标':<22}{'合成':>12}{'真实':>12}{'合成/真实':>12}   说明")
    print("-" * 100)
    for k, cn, s, r, ratio, why in lines:
        print(f"{cn:<22}{s:>12.5f}{r:>12.5f}{ratio:>12.3f}   {why}")
    print(f"\n→ 明细 JSON: {os.path.join(args.out, 'gap_report.json')}")


if __name__ == "__main__":
    main()
