#!/usr/bin/env python3
"""apply_ao.py — 把 C1 的 ao / contact 合成进 beauty（C2）

为什么要单独一个脚本，而不是塞进 procgen2：
    合成是**像素级后处理**，与 SVG 编译无关。独立脚本可以反复调参而不重渲染，
    也保证 B2 的渲染路径零改动（回归风险最小）。

合成公式（全部可调，默认值给了保守档）：
    shade = clamp(1 - ao_s*(1-ao), lo, 1)
          * clamp(1 - ct_s*(1-contact), lo2, 1)
    img'  = img * shade * gain          # gain 补偿整体压暗，保持平均亮度
可选假倒影（--wet > 0）：
    img[ground] = mix(img[ground], mirror+blur(img), wet * depth_weight)
    ← 沿地平线镜像 + 垂直拉长的高斯，模拟湿沥青的垂直光带（非射线追踪）

用法：
    .venv/bin/python apply_ao.py beauty.png --zdir ZDIR --out out.png \
        --ao 0.55 --contact 0.35 --wet 0.3 --report

    # 关掉某一项
    .venv/bin/python apply_ao.py b.png --zdir Z --out o.png --ao 0
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image, ImageFilter


def load_rgb(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0


def save_rgb(a, p):
    Image.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)).save(p)


def ground_mask_from_ids(zdir):
    """ids.npy + zmeta.id_kinds → 地面掩码。"""
    ids = np.load(os.path.join(zdir, "ids.npy"))
    meta = json.load(open(os.path.join(zdir, "zmeta.json"), encoding="utf-8"))
    kinds = meta.get("id_kinds", {})
    gids = [int(k) for k, v in kinds.items() if v == "ground"]
    if not gids:
        return np.zeros(ids.shape, dtype=bool), ids
    return np.isin(ids, gids), ids


def tail_down(a, start, decay=0.94, mode="max"):
    """单向向下拖尾。

    mode="max"：tail[y] = max(a[y], decay*tail[y-1])——现行默认（C2 实测 +29.5% 梯度）。
    mode="ema"：tail[y] = (1-decay)*a[y] + decay*tail[y-1]——审核轮 1 建议。
        max 是形态学膨胀：单像素亮点会被拉成笔直亮线（倒影竖条），1.5px 模糊盖不住。
        EMA 是线性低通，不会把孤立亮点拉成线；代价是倒影更"糊"，需 A/B 确认梯度与 rd。

    为什么不用对称模糊：模糊把亮点"稀释"，实测地面垂直梯度只 +2~7%；
    湿地面倒影是**光带**——亮点沿 y 拖出 30–50px 的尾巴。逐行递推即可。
    只从 start（地平线附近）开始遍历，省掉天空那 60% 的行。
    """
    out = np.zeros_like(a)
    acc = np.zeros_like(a[0])
    for y in range(max(start, 0), a.shape[0]):
        if mode == "ema":
            acc = (1.0 - decay) * a[y] + decay * acc
        else:
            acc = np.maximum(a[y], acc * decay)
        out[y] = acc
    return out


def fake_ssr(img, gmask, wet=0.4, decay=0.94, hi_thr=0.26, tail_mode="max",
             wet_base=0.15, wet_range=None):
    """简化倒影：建筑亮部沿地平线镜像 → 单向向下拖尾 → 按距离衰减 → **叠加**。

    ⚠️ 两条实测教训：
    1) 用"混合"会把地面原有纹理抹平（梯度 −4.2%）——湿地面是「原地面 + 反射光」，加法；
    2) 用对称模糊拉不出光带（梯度只 +2~7%）——必须**单向拖尾**。
    """
    h, w, _ = img.shape
    ys = np.where(gmask.any(axis=1))[0]
    if ys.size == 0:
        return img
    horizon = int(ys.min())
    src_y = 2 * horizon - np.arange(h)
    refl = np.zeros_like(img)
    ok = (src_y >= 0) & (src_y < max(horizon, 1))
    refl[ok] = img[src_y[ok]]
    refl[~ok] = img[min(max(horizon - 1, 0), h - 1)]

    luma = refl.mean(axis=2, keepdims=True)
    hi = np.clip((luma - hi_thr) / max(1.0 - hi_thr, 1e-6), 0.0, 1.0)
    refl_hi = refl * hi
    tail = tail_down(refl_hi, horizon - 8, decay=decay, mode=tail_mode)
    # 水平轻模糊：真实湿地面反射是光带，不是"扫描线"（建筑的窗户阵列会被镜像成竖条纹）
    tail = np.asarray(Image.fromarray((np.clip(tail, 0, 1) * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(1.5)), dtype=np.float32) / 255.0

    t = np.clip((np.arange(h) - horizon) / max(h - horizon, 1), 0, 1)
    # ⚠️ 权重必须随 t **递减**：画面底部的地面反射的是头顶天空，
    # 只有地平线附近的地面才反射建筑。反过来（底部最强）会让倒影一直拖到
    # 画面最下方，看起来像"贴了一层半透明的楼"。
    # V7：底部残留 0.15 会让倒影一路拖到画面最下方，与近岸的树/栏杆重叠。
    # wet_base 默认 0.15 保持旧行为；要"空间正确"就传 0，并配合 wet_range 限区间。
    w_row = 0.85 * (1.0 - t) ** 1.2 + wet_base
    if wet_range is not None:
        yy = np.arange(h) / max(h - 1, 1)
        # 边界必须渐变：硬截断会在水面留下一条"倒影结束线"（像被切开的水面）
        y0, y1 = wet_range
        fade_in = np.clip((yy - y0) / 0.04, 0.0, 1.0)
        fade_out = np.clip((y1 - yy) / 0.10, 0.0, 1.0)
        w_row = w_row * fade_in * fade_out
    amt = (wet * w_row)[:, None, None] * gmask[..., None]
    return np.clip(img + tail * amt, 0, 1)


def composite(beauty, zdir, ao_s=0.55, ct_s=0.35, wet=0.0,
              ao_lo=0.55, gain_mode="auto", wet_decay=0.94, shade_blur=0.0,
              wet_tail="max", contact_soft=0.30, haze_pp=0.0, godray=0.0,
              wet_base=0.15, wet_range=None,
              haze_color=(0.82, 0.88, 0.93), haze_near=0.15, haze_far=0.95,
              sun=(0.20, 0.15), godray_power=3.0, godray_radius=0.55):
    img = load_rgb(beauty)
    h, w, _ = img.shape
    ao = np.asarray(Image.open(os.path.join(zdir, "ao.png")).convert("L"),
                    dtype=np.float32) / 255.0
    ct = np.asarray(Image.open(os.path.join(zdir, "contact.png")).convert("L"),
                    dtype=np.float32) / 255.0
    if ao.shape != (h, w):
        ao = np.asarray(Image.fromarray((ao * 255).astype(np.uint8)).resize((w, h)),
                        dtype=np.float32) / 255.0
        ct = np.asarray(Image.fromarray((ct * 255).astype(np.uint8)).resize((w, h)),
                        dtype=np.float32) / 255.0
    gmask, ids = ground_mask_from_ids(zdir)

    # 柔和接触阴影：把 contact 大范围模糊一次，让楼脚/物体交接处形成 20–30px 的暗带
    ct_soft = np.asarray(Image.fromarray(((1 - ct) * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(14)), dtype=np.float32) / 255.0
    shade = np.clip(1 - ao_s * (1 - ao), ao_lo, 1.0) * \
            np.clip(1 - ct_s * (1 - ct), 1 - 0.5 * ct_s, 1.0) * \
            (1.0 - contact_soft * ct_soft)
    # ⚠️ contact_soft 是独立于 ct_s 的"大范围柔化接触"项：ct_s=0 时它仍生效。
    # C2.1 消融必须把它一起归零，否则"无接触阴影"组并不干净。
    # 柔化 shade：AO 的边缘是 1px 硬边（矩形楼的轮廓），会把 step_ratio 抬上去、
    # orient_entropy 压下去——实测 -8.5% 的收益里这三项是负的，柔化 0.5px 可回收一部分。
    if shade_blur > 0:
        shade = np.asarray(Image.fromarray((shade * 255).astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(shade_blur)), dtype=np.float32) / 255.0
    out = img * shade[..., None]

    # 保持平均亮度：AO 会让整图变暗，不补偿的话 chroma/flat 指标会连带恶化
    if gain_mode == "auto":
        m0, m1 = float(img.mean()), float(out.mean())
        gain = float(np.clip(m0 / max(m1, 1e-6), 1.0, 1.25))
    else:
        gain = float(gain_mode)
    out = np.clip(out * gain, 0, 1)

    if wet > 0:
        out = fake_ssr(out, gmask, wet=wet, decay=wet_decay, tail_mode=wet_tail,
                       wet_base=wet_base, wet_range=wet_range)

    # E2：空气透视（per-pixel depth）+ God ray（可选）
    dn = load_depth01(zdir)
    snr_before = std_far_near(out.mean(axis=2), dn) if dn is not None else None
    haze_map = None
    if haze_pp > 0:
        if dn is None:
            raise SystemExit("--haze-pp 需要 zdir 里有 depth.png")
        out, haze_map = apply_haze_pp(out, dn, haze_pp, haze_color, haze_near, haze_far)
    snr_after = std_far_near(out.mean(axis=2), dn) if dn is not None else None
    gr_map = None
    if godray > 0:
        if dn is None:
            raise SystemExit("--godray 需要 zdir 里有 depth.png")
        out, gr_map = aplicar_godray(out, dn, godray, sun, power=godray_power,
                                     radius=godray_radius)

    stats = {
        "ao_strength": ao_s, "contact_strength": ct_s, "wet": wet,
        "contact_soft": contact_soft,
        "wet_tail": wet_tail,
        "gain": round(gain, 4),
        "mean_before": round(float(img.mean()), 4),
        "mean_after": round(float(out.mean()), 4),
        "ground_frac": round(float(gmask.mean()), 4),
        "shade_mean": round(float(shade.mean()), 4),
        "shade_p05": round(float(np.percentile(shade, 5)), 4),
        "haze_pp": haze_pp, "godray": godray,
        "std_far_near_before": round(snr_before, 4) if snr_before else None,
        "std_far_near_after": round(snr_after, 4) if snr_after else None,
        "haze_map_p95": round(float(np.percentile(haze_map, 95)), 4) if haze_map is not None else None,
    }
    return out, stats, gmask


def load_depth01(zdir):
    """zbuf 的 depth.png：0=近、1=远（已 log 归一化）。"""
    p = os.path.join(zdir, "depth.png")
    if not os.path.exists(p):
        return None
    return np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255.0


def std_far_near(lum, dn, far_q=80, near_q=20):
    """空气透视的可测形式：远景/近景的局部对比度比（08 §12 #2）。"""
    f = dn >= np.percentile(dn, far_q)
    n = dn <= np.percentile(dn, near_q)
    if not f.any() or not n.any():
        return None
    sn = float(lum[n].std())
    return float(lum[f].std() / sn) if sn > 1e-6 else None


def apply_haze_pp(img, dn, strength=0.18, color=(0.82, 0.88, 0.93),
                  near=0.15, far=0.95, gamma=1.15):
    """逐像素空气透视（E2）：按 zbuf depth 查表混合，取代图层级的常数 haze。

    与 procgen2 的图层 haze 的区别：后者按 SVG 图元的 depth（每层一个数），
    同层的近处地面与远处楼脚被同等雾化；这里按 per-pixel depth，雾随距离单调增长。
    """
    f = np.clip((dn - near) / max(far - near, 1e-6), 0, 1) ** gamma
    c = np.asarray(color, dtype=np.float32)
    out = img * (1 - strength * f[..., None]) + c * strength * f[..., None]
    return np.clip(out, 0, 1), f


def aplicar_godray(img, dn, strength=0.12, sun=(0.20, 0.15), steps=6,
                   thr=0.72, scale=0.03, power=3.0, radius=0.55):
    """God ray 低成本近似：亮部沿"光源中心"做多尺度径向重采样后叠加。

    不是散射积分：没有遮挡判断，所以只让亮部（lum>thr）且偏远景（dn 小）参与，
    避免给近景楼面糊一层光。
    """
    if strength <= 0:
        return img, np.zeros(img.shape[:2], dtype=np.float32)
    lum = img.mean(axis=2)
    m = np.clip((lum - thr) / max(1 - thr, 1e-6), 0, 1) * np.clip(1 - dn * 2.0, 0, 1)
    m = m ** max(1.0, power)   # 限制在最亮区（太阳附近），避免整片天空被抬平
    h, w = lum.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = sun[0] * w, sun[1] * h
    dist = np.sqrt(((xx - cx) / w) ** 2 + ((yy - cy) / h) ** 2)
    m = m * np.clip(1.0 - dist / max(radius, 1e-6), 0.0, 1.0)   # 只保留太阳附近
    acc = np.zeros_like(img)
    for k in range(1, steps + 1):
        s = 1.0 + scale * k
        xi = np.clip(np.rint(cx + (xx - cx) / s).astype(np.int32), 0, w - 1)
        yi = np.clip(np.rint(cy + (yy - cy) / s).astype(np.int32), 0, h - 1)
        acc += img[yi, xi]
    acc /= steps
    out = np.clip(img + strength * acc * m[..., None], 0, 1)
    return out, m


def ground_vert_energy(img, gmask):
    """地面掩码内的垂直梯度能量（倒影验收判据：应上升 ≥20%）。"""
    g = img.mean(axis=2)
    dy = np.abs(np.diff(g, axis=0))
    m = gmask[1:] & gmask[:-1]
    return float(dy[m].mean()) if m.any() else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("beauty")
    ap.add_argument("--zdir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ao", type=float, default=0.55)
    ap.add_argument("--contact", type=float, default=0.35)
    ap.add_argument("--wet", type=float, default=0.0, help="湿地面倒影强度，0.4 约 +30% 垂直梯度")
    ap.add_argument("--wet-decay", type=float, default=0.94)
    ap.add_argument("--wet-tail", choices=("max", "ema"), default="max",
                    help="倒影拖尾算子：max=现行（可能拉出竖条）/ ema=线性低通（审核轮 1 建议）")
    ap.add_argument("--ao-lo", type=float, default=0.55)
    ap.add_argument("--shade-blur", type=float, default=0.0,
                    help="柔化 AO 边缘；实测 0.5px 无统计收益（+0.1%），默认关")
    ap.add_argument("--contact-soft", type=float, default=0.30,
                    help="大范围接触柔化权重；C2.1 消融时对应组设 0")
    ap.add_argument("--haze-pp", type=float, default=0.0,
                    help="E2：逐像素空气透视强度（0=关；建议 0.12-0.22）")
    ap.add_argument("--haze-color", default="#d1dee8")
    ap.add_argument("--haze-near", type=float, default=0.15)
    ap.add_argument("--haze-far", type=float, default=0.95)
    ap.add_argument("--wet-base", type=float, default=0.15,
                    help="倒影在画面底部的残留量；0=倒影随距离衰减到无（空间正确）")
    ap.add_argument("--wet-range", default=None,
                    help="y0,y1（0-1）：倒影只在该纵向区间生效，例如 0.40,0.62")
    ap.add_argument("--godray", type=float, default=0.0,
                    help="E2：God ray 强度（0=关；建议 0.06-0.12）")
    ap.add_argument("--godray-power", type=float, default=3.0,
                    help="God ray 亮部掩码指数（越大越只作用于太阳附近）")
    ap.add_argument("--godray-radius", type=float, default=0.55,
                    help="God ray 作用半径（画面归一化；0=不限）")
    ap.add_argument("--sun", default="0.20,0.15",
                    help="God ray 光源位置（画面归一化坐标 x,y）")
    ap.add_argument("--gain", default="auto")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    hc = args.haze_color.lstrip("#")
    hc = tuple(int(hc[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    sx, sy = (float(v) for v in args.sun.split(","))
    out, stats, gmask = composite(args.beauty, args.zdir, args.ao, args.contact,
                                  args.wet, args.ao_lo, args.gain,
                                  wet_decay=args.wet_decay, shade_blur=args.shade_blur,
                                  wet_tail=args.wet_tail, contact_soft=args.contact_soft,
                                  haze_pp=args.haze_pp, godray=args.godray,
                                  wet_base=args.wet_base,
                                  wet_range=(tuple(float(v) for v in args.wet_range.split(","))
                                             if args.wet_range else None),
                                  haze_color=hc, haze_near=args.haze_near,
                                  haze_far=args.haze_far, sun=(sx, sy),
                                  godray_power=args.godray_power,
                                  godray_radius=args.godray_radius)
    save_rgb(out, args.out)
    if args.report:
        before = load_rgb(args.beauty)
        e0 = ground_vert_energy(before, gmask)
        e1 = ground_vert_energy(out, gmask)
        stats["ground_vert_energy_before"] = round(e0, 5)
        stats["ground_vert_energy_after"] = round(e1, 5)
        stats["ground_vert_energy_delta_pct"] = round(100 * (e1 - e0) / max(e0, 1e-9), 1)
        print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"→ {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
