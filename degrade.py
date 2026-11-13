#!/usr/bin/env python3
"""degrade.py — 成像链「劣化」后处理：把矢量的完美感压回照片的统计域

设计原则
--------
真实照片之所以「像照片」，不是因为内容，而是因为**成像链在像素上留下了痕迹**：
   镜头 → 传感器 → ISP → JPEG → 平台二次压缩
矢量图把这一整条链都跳过了，所以它干净得可疑。

本脚本按上述链条的**逆序**逐级回放这些痕迹（顺序错会互相抵消）：
   1 optical_blur      镜头 MTF：边缘展宽（矢量是 1px 台阶）
   2 chromatic_aberration  径向色差：R/B 通道半径不一致（边缘起彩边）
   3 sensor_noise      散粒+读出噪声：暗部更强，带轻微色度分量
   4 chroma_subsample  ISP 前色度 4:2:0 下采样（真实 JPEG 一定有）
   5 low_freq_grime    低频脏纹理：镜头/传感器不均匀、灰尘、暗角不均
   6 tonemap           黑位抬升 + 轻微 S 曲线（真实黑场不是 0）
   7 vignette          余弦四次方暗角
   8 bloom             高光溢出（镜头雾/传感器饱和）
   9 jpeg              JPEG 二次压缩（quality 88-93，4:2:0）
  10 dither            8bit 抖动，避免渐变带状

用法
----
    # 单张
    python degrade.py in.png out.jpg --strength 1.0 --report
    # 批量（对目录里所有 png）
    python degrade.py --batch assets/ out/ --strength 0.8
    # 三联对比图（原图 | 劣化 | 真实）
    python degrade.py in.png out.jpg --sheet sheet.png --real ref.jpg

纪律：一次只调一个参数。`--strength` 只做全局缩放，归因时先把它固定在 1.0。
"""
import argparse
import glob
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEFAULTS = dict(
    blur_px=1.20,        # 1 光学模糊半径（像素）
    ca_k=0.0018,         # 2 色差系数（0.18% 半径差）
    noise_luma=0.024,    # 3 亮度噪声 σ（0-1 尺度）
    noise_chroma=0.014,  # 3 色度噪声 σ
    chroma_sub=True,     # 4 色度 4:2:0
    grime=0.020,         # 5 低频脏纹理幅度
    micro=0.055,         # 5b 材质微观起伏（1/f 多尺度噪声）
    sharpen=0.45,        # 5c ISP 的 USM 锐化（与 blur 成对，否则功率谱斜率会过头）
    black_lift=0.030,    # 6 黑位抬升
    s_curve=0.06,        # 6 S 曲线强度
    vignette=0.14,       # 7 暗角
    bloom=0.05,          # 8 高光溢出
    jpeg=89,             # 9 JPEG 质量（0=不压缩）
    dither=True,         # 10 抖动
    grain_seed=20261026,
)

# 输入类型分档：
# vector = 矢量/渲染图（无纹理、无噪声）——上面的默认值
# photo  = 扩散输出（已经有纹理与降噪）——必须把「补纹理」调小、「补噪声」调大，
#          否则 micro/grime 会与模型自带纹理叠乘，中间调直接过冲 5 倍。
PRESETS = {
    "vector": {},
    "photo": dict(
        # 扩散输出 = vector 档 × 0.75 的等效参数（实测最优，dist 0.372）
        micro=0.041, grime=0.015,
        noise_luma=0.018, noise_chroma=0.011,
        blur_px=0.90, ca_k=0.0014, jpeg=91,
    ),
}


# ------------------------------------------------------------------ 工具
def _bilinear(ch, xs, ys):
    h, w = ch.shape
    x0 = np.clip(np.floor(xs).astype(np.int32), 0, w - 1)
    y0 = np.clip(np.floor(ys).astype(np.int32), 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    fx = np.clip(xs - x0, 0, 1)
    fy = np.clip(ys - y0, 0, 1)
    return (
        ch[y0, x0] * (1 - fx) * (1 - fy)
        + ch[y0, x1] * fx * (1 - fy)
        + ch[y1, x0] * (1 - fx) * fy
        + ch[y1, x1] * fx * fy
    )


def _radial_scale(ch, k):
    """以画面中心为原点，按 (1+k) 缩放半径 → 模拟倍率色差。"""
    h, w = ch.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = (w - 1) * 0.5, (h - 1) * 0.5
    s = 1.0 + k
    return _bilinear(ch, cx + (xx - cx) / s, cy + (yy - cy) / s)


def _blur(a, radius):
    if radius <= 1e-4:
        return a
    im = Image.fromarray(np.clip(a * 255 + 0.5, 0, 255).astype(np.uint8))
    im = im.filter(ImageFilter.GaussianBlur(radius))
    return np.asarray(im, dtype=np.float32) / 255.0


# ------------------------------------------------------------------ 各工序
def optical_blur(a, p):
    return _blur(a, p["blur_px"])


def chromatic_aberration(a, p):
    if p["ca_k"] <= 0:
        return a
    out = a.copy()
    out[..., 0] = _radial_scale(a[..., 0], p["ca_k"])
    out[..., 2] = _radial_scale(a[..., 2], -p["ca_k"])
    return out


def sensor_noise(a, p, rng):
    luma = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    # 暗部噪声更明显（读出噪声主导）：0.4 + 1.2*(1-luma)
    weight = 0.4 + 1.2 * (1.0 - luma)
    nl = rng.normal(0, p["noise_luma"], size=luma.shape).astype(np.float32) * weight
    nc = rng.normal(0, p["noise_chroma"], size=a.shape).astype(np.float32) * weight[..., None]
    return a + nl[..., None] * 0.85 + nc


def chroma_subsample(a, p):
    if not p["chroma_sub"]:
        return a
    y = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    cb = -0.168736 * a[..., 0] - 0.331264 * a[..., 1] + 0.5 * a[..., 2]
    cr = 0.5 * a[..., 0] - 0.418688 * a[..., 1] - 0.081312 * a[..., 2]
    h, w = y.shape

    def half(ch):
        im = Image.fromarray(np.clip(ch * 0.5 + 0.5, 0, 1).astype(np.float32), mode="F")
        im = im.resize((max(2, w // 2), max(2, h // 2)), Image.Resampling.BILINEAR)
        im = im.resize((w, h), Image.Resampling.BILINEAR)
        return np.asarray(im, dtype=np.float32) * 2.0 - 1.0

    cb, cr = half(cb), half(cr)
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    return np.stack([r, g, b], axis=-1)


def low_freq_grime(a, p, rng):
    if p["grime"] <= 0:
        return a
    h, w = a.shape[:2]
    small = rng.normal(0, 1, size=(6, 6)).astype(np.float32)
    im = Image.fromarray(small, mode="F").resize((w, h), Image.Resampling.BICUBIC)
    f = np.array(im, dtype=np.float32)
    f /= np.abs(f).max() + 1e-9
    # 只作用在亮度上，且强度随亮度轻微变化（模拟不均匀照明）
    return a + (f * p["grime"])[..., None] * (0.5 + 0.5 * a)


def micro_texture(a, p, rng):
    """材质微观起伏：多尺度 1/f 噪声（亮度分量 + 三通道独立色度分量）。

    为什么必须有这一道：真实表面的 albedo 在 1/f 谱上持续有能量起伏，
    而矢量图的纯色块在这一层是“零”。这一个工序能同时抬高
    unique_color_ratio / grad_midtone_ratio / hue_entropy，并把 grad_kurtosis 拉低。
    """
    amp = p["micro"]
    if amp <= 0:
        return a
    h, w = a.shape[:2]

    def field(chan_rng):
        acc = np.zeros((h, w), np.float32)
        wsum = 0.0
        for o in range(3, 9):
            s = 2 ** o
            if s > min(h, w):
                break
            g = chan_rng.normal(0, 1, (s, s)).astype(np.float32)
            im = Image.fromarray(g, mode="F").resize((w, h), Image.Resampling.BICUBIC)
            acc += np.array(im, dtype=np.float32) * (1.0 / (o - 2))
            wsum += 1.0 / (o - 2)
        acc /= wsum
        return acc / (np.abs(acc).max() + 1e-9)

    luma = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    shine = (0.35 + 0.65 * luma)[..., None]
    out = a + (field(rng) * amp)[..., None] * shine
    for c in range(3):
        out[..., c] += field(rng) * (amp * 0.55)
    return out


def isp_sharpen(a, p):
    """ISP 的 USM 锐化：真实相机在「光学模糊」之后必然做这一步，
    所以功率谱斜率才能回到 ≈-2 而不牺牲边缘过渡宽度。"""
    amt = p["sharpen"]
    if amt <= 0:
        return a
    u8 = (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)
    im = Image.fromarray(u8).filter(
        ImageFilter.UnsharpMask(radius=1.6, percent=int(amt * 100), threshold=2)
    )
    return np.asarray(im, dtype=np.float32) / 255.0


def tonemap(a, p):
    x = np.clip(a, 0, 1)
    if p["s_curve"] > 0:
        s = p["s_curve"]
        x = x + s * (x - 0.5) * x * (1 - x) * 4.0
    x = p["black_lift"] + (1.0 - p["black_lift"]) * x
    return np.clip(x, 0, 1)


def vignette(a, p):
    if p["vignette"] <= 0:
        return a
    h, w = a.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = (w - 1) * 0.5, (h - 1) * 0.5
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / np.sqrt(cx ** 2 + cy ** 2)
    v = 1.0 - p["vignette"] * (r ** 2.1)
    return a * v[..., None]


def bloom(a, p):
    if p["bloom"] <= 0:
        return a
    luma = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    hi = np.clip((luma - 0.72) / 0.28, 0, 1)
    hi_rgb = a * hi[..., None]
    b = _blur(hi_rgb, max(2.0, min(a.shape[:2]) * 0.004))
    return np.clip(a + b * p["bloom"] * 2.0, 0, 1)


def encode_jpeg(a, p, dst):
    """最终落盘（只写一次）：q>0 时走 JPEG 4:2:0 真实编码，否则直接存 PNG。"""
    q = int(p["jpeg"])
    u8 = (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)
    if q <= 0:
        Image.fromarray(u8).save(dst)
        return a
    tmp = os.path.splitext(dst)[0] + ".__jpeg_tmp.jpg"
    Image.fromarray(u8).save(tmp, quality=q, subsampling=2, optimize=True)
    a2 = np.asarray(Image.open(tmp).convert("RGB"), dtype=np.float32) / 255.0
    if dst.lower().endswith(".png"):
        Image.fromarray((np.clip(a2, 0, 1) * 255 + 0.5).astype(np.uint8)).save(dst)
        os.remove(tmp)
    else:
        os.replace(tmp, dst)
    return a2


def dither(a, p, rng):
    if not p["dither"]:
        return a
    return a + rng.uniform(-0.5, 0.5, size=a.shape).astype(np.float32) / 255.0


# ------------------------------------------------------------------ 主链
def degrade(path, dst, params=None, seed=None):
    p = dict(DEFAULTS)
    if params:
        p.update(params)
    rng = np.random.default_rng(p["grain_seed"] if seed is None else seed)

    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    a = optical_blur(a, p)
    a = chromatic_aberration(a, p)
    a = sensor_noise(a, p, rng)
    a = chroma_subsample(a, p)
    a = low_freq_grime(a, p, rng)
    a = micro_texture(a, p, rng)
    a = isp_sharpen(a, p)
    a = tonemap(a, p)
    a = vignette(a, p)
    a = bloom(a, p)
    a = dither(a, p, rng)
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    encode_jpeg(a, p, dst)
    return dst


def make_sheet(syn, deg, real, out, labels=("SYNTHETIC", "DEGRADED", "REAL"), h=420):
    ims = []
    for p in (syn, deg, real):
        im = Image.open(p).convert("RGB")
        w = int(im.width * h / im.height)
        ims.append(im.resize((w, h), Image.Resampling.LANCZOS))
    pad, top = 10, 22
    W = sum(i.width for i in ims) + pad * (len(ims) + 1)
    canvas = Image.new("RGB", (W, h + top + pad), (24, 24, 28))
    d = ImageDraw.Draw(canvas)
    x = pad
    for im, lab in zip(ims, labels):
        canvas.paste(im, (x, top))
        d.text((x + 4, 6), lab, fill=(230, 230, 230))
        x += im.width + pad
    canvas.save(out)
    return out


# ------------------------------------------------------------------ CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="?")
    ap.add_argument("dst", nargs="?")
    ap.add_argument("--batch", nargs=2, metavar=("INDIR", "OUTDIR"))
    ap.add_argument("--strength", type=float, default=1.0,
                    help="全局强度缩放（1.0=预设默认，0=原图）")
    ap.add_argument("--preset", choices=list(PRESETS), default="vector",
                    help="vector=矢量图; photo=扩散输出（纹理参数更小、噪声更大）")
    ap.add_argument("--set", action="append", default=[], metavar="K=V",
                    help="单项覆盖，如 --set noise_luma=0.02")
    ap.add_argument("--sheet")
    ap.add_argument("--real")
    ap.add_argument("--report", action="store_true", help="打印劣化前后的指标对比")
    args = ap.parse_args()

    p = dict(DEFAULTS)
    p.update(PRESETS[args.preset])
    for kv in args.set:
        k, v = kv.split("=", 1)
        if k not in DEFAULTS:
            sys.exit(f"未知参数 {k}；可选：{', '.join(DEFAULTS)}")
        cur = DEFAULTS[k]
        p[k] = (v.lower() in ("true", "1")) if isinstance(cur, bool) else type(cur)(v)
    # strength 只缩放"痕迹类"参数，jpeg 质量例外
    s = args.strength
    for k in ("blur_px", "ca_k", "noise_luma", "noise_chroma", "grime", "micro",
              "sharpen", "black_lift", "s_curve", "vignette", "bloom"):
        p[k] = p[k] * s

    if args.batch:
        indir, outdir = args.batch
        files = sorted(glob.glob(os.path.join(indir, "*.png")))
        if not files:
            sys.exit(f"{indir} 下没有 png")
        for f in files:
            dst = os.path.join(outdir, os.path.splitext(os.path.basename(f))[0] + ".jpg")
            degrade(f, dst, p)
            print("→", dst)
        return

    if not args.src or not args.dst:
        sys.exit("需要 src dst，或用 --batch INDIR OUTDIR")
    degrade(args.src, args.dst, p)
    print("→", args.dst)

    if args.report:
        from analyze_gap import KEYS, measure
        a, b = measure(args.src), measure(args.dst)
        print(f"\n{'指标':<22}{'原图':>11}{'劣化后':>11}{'变化':>10}   真实基准参考")
        print("-" * 88)
        for k, cn, why in KEYS:
            va, vb = a[k], b[k]
            chg = (vb / va) if abs(va) > 1e-9 else float("nan")
            print(f"{cn:<22}{va:>11.5f}{vb:>11.5f}{chg:>10.2f}×   {why}")

    if args.sheet:
        real = args.real or args.src
        print("↳ 三联图:", make_sheet(args.src, args.dst, real, args.sheet))


if __name__ == "__main__":
    main()
