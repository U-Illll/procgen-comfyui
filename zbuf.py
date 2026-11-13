#!/usr/bin/env python3
"""zbuf.py — 场景图 → per-pixel 深度与可见性通道（C1）

为什么需要它（05 §11）：
    现在 depth.png 是「每个图元一个常数灰度」，图元之间没有遮挡关系。
    接 ControlNet 会把每栋楼理解成一张平纸片；也无法算 AO / 接触阴影 / 倒影。
    本脚本用 numpy 光栅化出真正的 z-buffer（1920×1080 约百毫秒），不改依赖。

输出（与 06 §4 的 G-buffer 同名同约定，可被 L4b/Cycles 的产物整体替换）：
    depth.png    近白远黑（MiDaS 约定，ControlNet depth 直接吃）
    normal.png   Sobel(depth) → 128+127·n
    ao.png       三档半径 SSAO（C2 用）
    contact.png  窄半径遮蔽 = 接触阴影（C2 用）
    seg.png      图元 id 上色（与现有 seg 通道一致）
    zbuf.npy     float32 米制 z（调试 / 后续 SSR）
    zmeta.json   参数与自检指标（台阶数 / 耗时 / 图元数）

用法：
    .venv/bin/python zbuf.py baseline/scenes/night_blue.json \
        --out ~/.cache/procgen/zb --time

    # 只算深度（跳过 AO，快 3-5 倍）
    .venv/bin/python zbuf.py scene.json --out DIR --no-ao
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from procgen2 import (BACKGROUND_KINDS, Compiler, depth_span,  # noqa: E402
                      rgb2hex, scene_from_dict)

NEAR, FAR = 0.6, 400.0          # 相机近/远裁剪（米，示意量级）
GROUND_FAR_K = 0.9              # 地平线处 z ≈ FAR*0.9（比任何楼都远）
GROUND_NEAR_K = 3.0             # 画面底部 z ≈ NEAR*3


# ---------------------------------------------------------------- 深度场

def view_z(d):
    """归一化 depth(0=远,1=近) → 米制 z（单调递减）。支持标量与数组。"""
    return FAR - np.asarray(d, dtype=np.float32) * (FAR - NEAR)


def ground_z(y0, y1, h):
    """地面：按透视（1/z 线性）从「地平线≈远裁剪」插到「画面底≈近」。

    不能直接用 depth_span：地面在现实中从脚下延伸到地平线，
    用常数 depth 会让地面整体比楼还远，AO 完全失效。
    """
    ys = np.arange(y0, y1, dtype=np.float32)
    t = np.clip((ys - y0) / max(y1 - y0 - 1, 1), 0.0, 1.0)
    z_h = FAR * GROUND_FAR_K
    z_b = NEAR * GROUND_NEAR_K
    inv = (1 - t) / z_h + t / z_b
    return 1.0 / np.maximum(inv, 1e-9)


def _shape_mask(shape, geom, x0, y0, bw, bh):
    """在 bbox 内生成真实轮廓 mask（None = 整块矩形）。

    只按 bbox 光栅化会把圆形树冠 / 梯形塔身写成方块——实测 depth.png 里
    树变成"方块堆"、塔身变方。这里用 PIL 在小图上画，开销只在 bbox 尺寸。
    """
    if shape == "rect" or geom is None or bw <= 0 or bh <= 0:
        return None
    mk = Image.new("1", (bw, bh), 0)
    d = ImageDraw.Draw(mk)
    if shape == "circle":
        cx, cy, r = geom
        d.ellipse([cx - r - x0, cy - r - y0, cx + r - x0, cy + r - y0], fill=1)
    elif shape == "poly":
        d.polygon([(px - x0, py - y0) for px, py in geom], fill=1)
    else:
        return None
    return np.asarray(mk, dtype=bool)


def rasterize(scene, drawn, w, h, ground_model="narrow"):
    """画家顺序 + per-pixel z-test。返回 (zbuf_m, idbuf)。"""
    zbuf = np.full((h, w), FAR, dtype=np.float32)
    idbuf = np.full((h, w), -1, dtype=np.int32)
    # id 必须是**节点**编号，不是绘制序号：一棵树会 emit 多个 lobe，
    # 用绘制序号会让同一棵树的叶子互相遮蔽（实测 AO 里出现同心圆环）。
    node_ids = {}
    node_kinds = {}
    node_boxes = {}

    for i, rec in enumerate(drawn):
        node, x0, y0, x1, y1, hint = rec[:6]
        nid = node_ids.setdefault(node.id, len(node_ids))
        node_kinds[nid] = node.kind
        if nid not in node_boxes:
            node_boxes[nid] = node.bbox or (float(rec[1]), float(rec[2]),
                                            float(rec[3]), float(rec[4]))
        shape = rec[6] if len(rec) > 6 else "rect"
        geom = rec[7] if len(rec) > 7 else None
        if node.kind in BACKGROUND_KINDS:
            continue                       # 天空/太阳/雾/暗角/颗粒不吃 z-test
        if x1 <= x0 or y1 <= y0:
            continue
        x0 = int(max(0, math.floor(x0)))
        y0 = int(max(0, math.floor(y0)))
        x1 = int(min(w, math.ceil(x1)))
        y1 = int(min(h, math.ceil(y1)))
        if x1 <= x0 or y1 <= y0:
            continue

        if node.kind == "ground" or hint == "ground":
            if ground_model == "perspective":
                # 真实透视地面：1/z 从地平线（GROUND_FAR_K·FAR）线性插到画面底（GROUND_NEAR_K·NEAR）。
                # ground_z() 早就写好但一直没接进光栅化（审核轮 1 指出的隐藏死代码）。
                zc = ground_z(y0, y1, h)
                sub = zbuf[y0:y1, x0:x1]
                subid = idbuf[y0:y1, x0:x1]
                zc2 = np.repeat(np.asarray(zc, dtype=np.float32)[:, None], x1 - x0, axis=1)
                m = (subid == -1) | (zc2 < sub)
                if m.any():
                    sub[m] = zc2[m]
                    subid[m] = nid
                continue
            # 旧模型：z 夹在 depth 附近的窄带（不能从 FAR 起算，否则透视会把
            # 画面底部压到 2m，把所有前景树都吞掉）；且**只填空像素**——
            # 地面是背景板，不该覆盖已经画好的楼。
            d = max(0.0, min(1.0, node.depth))
            z_h = view_z(max(0.0, d - 0.06))      # 地平线（远）
            z_b = view_z(min(1.0, d + 0.03))      # 画面底（近）
            ys = np.arange(y0, y1, dtype=np.float32)
            t = np.clip((ys - y0) / max(y1 - y0 - 1, 1), 0.0, 1.0)
            inv = (1 - t) / z_h + t / z_b
            zc = 1.0 / np.maximum(inv, 1e-9)
            sub = zbuf[y0:y1, x0:x1]
            subid = idbuf[y0:y1, x0:x1]
            zc2 = np.repeat(zc[:, None], x1 - x0, axis=1)
            m = (subid == -1) | (zc2 < sub)
            if m.any():
                sub[m] = zc2[m]
                subid[m] = nid
            continue
        else:
            d0, d1 = depth_span(node, hint)
            ys = np.arange(y0, y1, dtype=np.float32)
            t = (ys - y0) / max(y1 - y0 - 1, 1)
            zc = view_z(d0 + (d1 - d0) * t)

        sub = zbuf[y0:y1, x0:x1]
        zc2 = np.repeat(zc[:, None], x1 - x0, axis=1)
        smask = _shape_mask(shape, geom, x0, y0, x1 - x0, y1 - y0)
        m = (zc2 < sub) if smask is None else ((zc2 < sub) & smask)
        if m.any():
            sub[m] = zc2[m]
            idbuf[y0:y1, x0:x1][m] = nid

    return zbuf, idbuf, node_kinds, node_ids, node_boxes


# ---------------------------------------------------------------- 派生通道

Z_FLOOR = NEAR * 0.5


def normalized_depth(z):
    """米制 z → 归一化深度（1=近，0=远）：**固定物理范围 + 对数 + gamma**。

    三次实测教训：
    1) 线性映射把 0–30m 的地面压成一片近白（整块 255）→ ControlNet 读成"一堵墙"；
    2) 固定范围的对数映射把楼压到 0.05–0.1（几乎全黑）→ 楼与天空不可分；
    3) 用**本图的实际分位**做 min-max（上一版）→ 跨景不一致：少一栋近景楼，
       整张图的 z 映射就跳变，与 ControlNet 需要的"稳定相对深度"冲突
       （审核轮 1 指出，属实）。
    定稿：范围固定为 [NEAR, FAR]（同一物理距离永远同一灰度），再开 gamma=0.7
    把楼体从 0.12 抬到 0.23（可见但仍明显暗于地面）。
    """
    lo, hi = math.log(NEAR), math.log(FAR)
    ln = np.log(np.clip(z, NEAR, FAR))
    d = 1.0 - (ln - lo) / (hi - lo)
    return np.clip(d, 0.0, 1.0) ** 0.7


def _lowpass(d, k=32):
    """低频近似（先降采样再上采样），用于把 AO 限制在「小尺度起伏」。"""
    h, w = d.shape
    sm = Image.fromarray(np.clip(d * 255, 0, 255).astype(np.uint8)).resize(
        (max(w // k, 1), max(h // k, 1)), Image.Resampling.BILINEAR)
    return np.asarray(sm.resize((w, h), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0


def normals_from_z(z, max_slope=5.0):
    """由**米制 z** 的梯度推法线，且在物体边界处退回正面 (0,0,1)。

    两个实测修正：
    1) 用归一化深度 d 算梯度会得到 ny≈0.002（G 通道 127.7）——地面法线等于没有，
       菲涅尔/反射拿不到"朝上"的信息；
    2) 直接用米制 z 又会在楼与楼边界（Δz≈100m/px）把边界当成斜面。
    所以：米制梯度 + 斜率阈值（>max_slope 视为不连续，法线保持正面）。
    """
    gx = np.zeros_like(z, dtype=np.float32)
    gy = np.zeros_like(z, dtype=np.float32)
    gx[:, 1:-1] = (z[:, 2:] - z[:, :-2]) * 0.5
    gy[1:-1, :] = (z[2:, :] - z[:-2, :]) * 0.5
    slope = np.sqrt(gx * gx + gy * gy)
    w = (slope < max_slope).astype(np.float32)          # 边界/跳变不参与
    nx, ny = -gx * w, -gy * w
    nz = np.ones_like(z, dtype=np.float32)
    ln = np.sqrt(nx * nx + ny * ny + nz * nz)
    return nx / ln, ny / ln, nz / ln


def ssao(z, idbuf, radii=(4, 16), taps=8, cap=0.15, hybrid=False, self_thr=0.08):
    """2.5D 卡片场景的 SSAO：只在**实体-实体交界**处算遮蔽。

    三次实测教训（每一条都在 ao.png 上肉眼可见）：
    1) 用裸深度差 → 地面自己的透视坡度把整片地面遮蔽（AO 覆盖 0.22 全在地面）；
    2) 去掉低频后仍有"浮雕"：楼体的层内梯度（span 0.045）被当成遮蔽，
       每栋楼边缘出现一圈亮边+暗边；
    3) 轮廓对天空也被算成遮蔽。
    正确判据：**邻居更近 且 邻居是实体 且 中心是实体 且 两者不是同一图元**。
    - 同图元内部（层内渐变）→ 0（消除浮雕）
    - 楼 vs 天空 → 0（天空不遮蔽）
    - 楼底 vs 地面 / 楼 vs 楼 → 有遮蔽（这才是接触阴影与暗角）
    """
    lz = np.log(np.maximum(z, Z_FLOOR))
    occ = np.zeros_like(lz)
    solid = idbuf >= 0
    for r in radii:
        for k in range(taps):
            a = 2 * math.pi * k / taps
            dx = int(round(r * math.cos(a)))
            dy = int(round(r * math.sin(a)))
            if dx == 0 and dy == 0:
                continue
            hn = np.roll(np.roll(lz, dy, 0), dx, 1)
            idn = np.roll(np.roll(idbuf, dy, 0), dx, 1)
            raw = lz - hn
            diff = np.clip(raw / cap, 0.0, 1.0)
            # hybrid：同图元内的**深度突变**（退台/屋顶设备/凹角）也算遮蔽。
            # self_thr 用 log 空间（0.08≈8% 深度突变），层内渐变 0.045 不触发 → 不产生浮雕。
            if hybrid:
                mask = solid & (idn >= 0) & ((idn != idbuf) | (raw > self_thr))
            else:
                mask = solid & (idn >= 0) & (idn != idbuf)
            occ += mask * diff
    occ /= max(len(radii) * taps, 1)
    return occ


def count_peaks(hist, rel=0.05, min_gap=1):
    """直方图峰数：用来判断「每个图元是否还能被分辨」（纸片判据的反面）。"""
    hmax = hist.max() if hist.size else 0
    if hmax <= 0:
        return 0
    thr = hmax * rel
    peaks, last = 0, -10 ** 9
    for i in range(1, len(hist) - 1):
        if hist[i] >= thr and hist[i] >= hist[i - 1] and hist[i] >= hist[i + 1]:
            if i - last >= min_gap:
                peaks += 1
                last = i
    return peaks


def colorize_ids(idbuf):
    """id → RGB（与 procgen2 的 seg 风格一致：乘质数散列）。"""
    out = np.zeros(idbuf.shape + (3,), dtype=np.uint8)
    ids = idbuf >= 0
    v = (idbuf[ids] + 1).astype(np.int32)
    out[..., 0][ids] = (v * 53) % 256
    out[..., 1][ids] = (v * 97) % 256
    out[..., 2][ids] = (v * 149) % 256
    return out


# ---------------------------------------------------------------- 主流程

def build(scene_path, outdir, size=None, with_ao=True, downsample_ao=1,
          ground_model="narrow", ssao_hybrid=False):
    t0 = time.time()
    d = json.load(open(scene_path, encoding="utf-8"))
    sc = scene_from_dict(d)
    if size:
        sc.w, sc.h = size
    w, h = sc.w, sc.h

    # seg 模式只为填 bbox 与 drawn（10 ms 级），不渲染 Chrome
    c = Compiler(sc, "seg")
    c.compile()
    drawn = c.drawn
    t_geom = time.time() - t0

    t1 = time.time()
    zbuf, idbuf, node_kinds, node_ids, node_boxes = rasterize(sc, drawn, w, h, ground_model=ground_model)
    t_rast = time.time() - t1

    dn = normalized_depth(zbuf)

    t2 = time.time()
    if with_ao:
        if downsample_ao > 1:
            hh, ww = h // downsample_ao, w // downsample_ao
            # ⚠️ 必须降采样**米制 z**，不是归一化深度：ssao 内部要取 log(z)，
            # 拿 log 空间的结果再 exp 回来等于把整条深度曲线搞歪（早期版本就错在这）。
            small_z = np.asarray(Image.fromarray(zbuf.astype(np.float32), mode="F").resize(
                (ww, hh), Image.Resampling.BILINEAR), dtype=np.float32)
            ids_s = np.asarray(Image.fromarray(
                ((idbuf + 1) % 255).astype(np.uint8)).resize((ww, hh), Image.Resampling.NEAREST),
                dtype=np.int32) - 1
            up = lambda a: np.asarray(Image.fromarray(a.astype(np.float32), mode="F").resize(
                (w, h), Image.Resampling.BILINEAR), dtype=np.float32)
            occ = up(ssao(small_z, ids_s, hybrid=ssao_hybrid))
            occ_narrow = up(ssao(small_z, ids_s, radii=(2,), taps=8, hybrid=ssao_hybrid))
        else:
            occ = ssao(zbuf, idbuf, hybrid=ssao_hybrid)
            # 接触阴影：只看窄半径（3px）的遮蔽 → 墙脚/物体交接处的暗带
            occ_narrow = ssao(zbuf, idbuf, radii=(3,), taps=8, hybrid=ssao_hybrid)
        ao = np.clip(1.0 - 1.6 * occ, 0.0, 1.0)
        contact = np.clip(1.0 - np.clip(occ_narrow * 8.0, 0.0, 1.0) ** 0.5, 0.0, 1.0)
    else:
        ao = np.ones_like(dn)
        contact = np.ones_like(dn)
    t_ao = time.time() - t2

    nx, ny, nz = normals_from_z(zbuf)
    os.makedirs(outdir, exist_ok=True)
    # compress_level=1：这五张是中间产物，压缩率不重要（默认 6 要 200ms+，占总耗时一半）
    _save = lambda a, n: Image.fromarray(a).save(os.path.join(outdir, n), compress_level=1)
    _save((dn * 255 + 0.5).astype(np.uint8), "depth.png")
    _save(np.stack([(128 + 127 * nx).clip(0, 255),
                    (128 + 127 * ny).clip(0, 255),
                    (128 + 127 * nz).clip(0, 255)], axis=-1).astype(np.uint8), "normal.png")
    _save((ao * 255 + 0.5).astype(np.uint8), "ao.png")
    _save((contact * 255 + 0.5).astype(np.uint8), "contact.png")
    _save(colorize_ids(idbuf), "seg.png")
    np.save(os.path.join(outdir, "zbuf.npy"), zbuf)
    np.save(os.path.join(outdir, "ids.npy"), idbuf)   # C2 的地面掩码靠它

    hist, _ = np.histogram(dn, bins=128, range=(0, 1))
    n_prims = sum(1 for n, *_ in drawn if n.kind not in BACKGROUND_KINDS)
    # 深度"可分辨性"判据：覆盖像素 >0.1% 的直方图 bin 数 vs 深度组数。
    # 纸片 = 每个图元一个常数灰度 → covered ≈ 组数；带层内梯度后每个组会摊到多 bin。
    covered = int((hist > 0.001 * dn.size).sum())
    groups = len({round(float(n.depth), 3) for n, *_ in drawn
                  if n.kind not in BACKGROUND_KINDS})
    meta = {
        "w": w, "h": h, "near": NEAR, "far": FAR,
        "n_drawn": len(drawn), "n_prims": n_prims,
        "depth_groups": groups,
        "depth_covered_bins": covered,
        "depth_peaks": int(count_peaks(hist)),
        "level_target": max(8, 3 * groups),
        "ao_coverage": round(float((ao < 0.9).mean()), 4),
        "contact_coverage": round(float((contact < 0.95).mean()), 4),
        "t_geom_ms": round(t_geom * 1000, 1),
        "t_raster_ms": round(t_rast * 1000, 1),
        "t_ao_ms": round(t_ao * 1000, 1),
        "t_total_ms": round((time.time() - t0) * 1000, 1),
        "id_kinds": {str(k): v for k, v in sorted(node_kinds.items())},
        "id_names": {str(v): k for k, v in sorted(node_ids.items(), key=lambda kv: kv[1])},
        "id_boxes": {str(k): [round(float(x), 1) for x in v] for k, v in sorted(node_boxes.items())},
    }
    json.dump(meta, open(os.path.join(outdir, "zmeta.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", help="WxH，覆盖场景分辨率")
    ap.add_argument("--no-ao", action="store_true")
    ap.add_argument("--ao-down", type=int, default=2,
                    help="AO 降采样倍数（默认 2=半分辨率；全分辨率实测 547–587ms 超预算）")
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--ssao-hybrid", action="store_true",
                    help="同图元深度突变也算遮蔽（C3 实验；默认关）")
    ap.add_argument("--ground-model", choices=("narrow", "perspective"), default="narrow",
                    help="地面深度模型：narrow=现行 depth±0.06 窄带 / perspective=1/z 透视斜面（审核轮 1 建议）")
    args = ap.parse_args()

    size = None
    if args.size:
        w, h = args.size.lower().split("x")
        size = (int(w), int(h))

    meta = build(args.scene, args.out, size=size,
                 with_ao=not args.no_ao, downsample_ao=max(1, args.ao_down),
                 ground_model=args.ground_model, ssao_hybrid=args.ssao_hybrid)
    print(f"prims={meta['n_prims']} drawn={meta['n_drawn']} 深度组={meta['depth_groups']}")
    print(f"深度可分辨：covered_bins={meta['depth_covered_bins']} "
          f"(目标 ≥{meta['level_target']}) peaks={meta['depth_peaks']}")
    print(f"AO 覆盖={meta['ao_coverage']} 接触阴影覆盖={meta['contact_coverage']}")
    print(f"耗时 几何={meta['t_geom_ms']}ms 光栅={meta['t_raster_ms']}ms "
          f"AO={meta['t_ao_ms']}ms 总={meta['t_total_ms']}ms -> {args.out}")
    ok = meta["depth_covered_bins"] >= meta["level_target"]
    print(f"深度连续性判据：{'PASS' if ok else 'FAIL'}"
          f"（纸片时 covered≈组数，见 08 §9.4 的「5~15 台阶 → 192 级」）")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
