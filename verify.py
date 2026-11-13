#!/usr/bin/env python3
"""verify.py — 三轨验收工具（A1 阶段：先实现统计轨与"距离口径"）

背景（见 Desktop/等效生图-程序化绘图/10-工程实施路线-20261026.md 的 A1）：
    08 文档里的"平均相对距离"没有代码实现，只有逐项 ratio。这里把它固化成
    两个口径并写进报告，之后所有对比都用同一口径。

口径定义（都基于逐项 ratio = synthetic / real）：
    rd_r1   = mean(|ratio - 1|)            线性相对偏差（对 2x 与 0.5x 对称）
    rd_log2 = mean(|log2(ratio)|)           对数相对偏差（对量纲更稳，重尾被压缩）

用法：
    # 只固化口径（读已有报告，写回 rel_dist 字段）
    .venv/bin/python verify.py --report baseline/gap_report.json

    # 分组复算（模拟"只用写实向场景"的基线）
    .venv/bin/python verify.py --report baseline/gap_report.json --include "shenzhen|style_day|night_blue|dusk_warm"

    # 两次报告对比（改动的效果）
    .venv/bin/python verify.py --compare baseline/gap_report.json /tmp/b2/gap_report.json

    # 直接测（不落盘）
    .venv/bin/python verify.py --syn "OUT/*_beauty.png" --real "REF/*.jpg"
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_gap as AG          # 复用 13 指标（KEYS / measure）


# ---------------------------------------------------------------- 口径

def _avg(rows, k):
    vals = [r[k] for r in rows if isinstance(r.get(k), (int, float)) and math.isfinite(r[k])]
    return float(np.mean(vals)) if vals else float("nan")


def build_comparison(S, R, keys=None):
    """与 analyze_gap.main 的 comparison 同构。"""
    keys = keys or [k for k, _, _ in AG.KEYS]
    out = []
    for k in keys:
        s, r = _avg(S, k), _avg(R, k)
        ratio = (s / r) if abs(r) > 1e-9 else float("nan")
        out.append({"key": k, "synthetic": round(s, 5), "real": round(r, 5),
                    "ratio": round(ratio, 3)})
    return out


def rel_dist(cmp):
    """两个口径 + 中位数版本。ratio 非有限值一律剔除并记数。"""
    r1, lg = [], []
    for c in cmp:
        ratio = c.get("ratio")
        if not isinstance(ratio, (int, float)) or not math.isfinite(ratio) or ratio <= 0:
            continue
        r1.append(abs(ratio - 1.0))
        lg.append(abs(math.log2(ratio)))
    if not r1:
        return {"n_used": 0}
    return {
        "n_used": len(r1),
        "rd_r1_mean": round(float(np.mean(r1)), 4),
        "rd_r1_median": round(float(np.median(r1)), 4),
        "rd_log2_mean": round(float(np.mean(lg)), 4),
        "rd_log2_median": round(float(np.median(lg)), 4),
        # 契约：后续一律用 rd_r1_mean 报分（与 08 的 0.845/0.720 数量级一致）
        "primary": "rd_r1_mean",
    }


def worst(cmp, n=5):
    rows = [c for c in cmp if isinstance(c.get("ratio"), (int, float))
            and math.isfinite(c["ratio"]) and c["ratio"] > 0]
    rows.sort(key=lambda c: -abs(math.log2(c["ratio"])))
    return rows[:n]


# ---------------------------------------------------------------- 报告处理

def load_report(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def subset(report, pattern):
    """按 file 名正则过滤合成集后重算 comparison（真实集不动）。"""
    import re
    rx = re.compile(pattern)
    S = [r for r in report["synthetic"] if rx.search(r.get("file", ""))]
    R = report["real"]
    if not S:
        sys.exit(f"--include {pattern!r} 过滤后合成集为空")
    keys = [k for k, _, _ in AG.KEYS]
    cmp = build_comparison(S, R, keys)
    # 保留中文名与说明
    meta = {c["key"]: c for c in report.get("comparison", [])}
    for c in cmp:
        if c["key"] in meta:
            c["name"] = meta[c["key"]].get("name", c["key"])
            c["why"] = meta[c["key"]].get("why", "")
    return {"synthetic": S, "real": R, "comparison": cmp,
            "subset": pattern, "n_synthetic": len(S)}


def report_line(tag, rd):
    return (f"{tag:<34} rd_r1(mean)={rd.get('rd_r1_mean'):<7} "
            f"median={rd.get('rd_r1_median'):<7} "
            f"rd_log2(mean)={rd.get('rd_log2_mean'):<7} n={rd.get('n_used')}")


def emit(rep, out_path=None, tag="current"):
    rd = rel_dist(rep["comparison"])
    print(report_line(tag, rd))
    print("\n最差的 5 项（按 |log2 ratio| 排序，>1 表示合成偏高）：")
    for c in worst(rep["comparison"], 5):
        print(f"  {c.get('name', c['key']):<22} syn={c['synthetic']:<10} "
              f"real={c['real']:<10} ratio={c['ratio']:<8} {c.get('why','')}")
    if out_path:
        rep["rel_dist_meter"] = ("rd_r1_mean = mean(|synthetic/real - 1|)；"
                                 "rd_log2 仅作交叉核对，不用于报分")
        rep["rel_dist"] = rd
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)
        print(f"\n→ 已写回 {out_path}（字段 rel_dist / rel_dist_meter）")
    return rd


def compare(a_path, b_path):
    A, B = load_report(a_path), load_report(b_path)
    rdA, rdB = rel_dist(A["comparison"]), rel_dist(B["comparison"])
    print(report_line(os.path.basename(a_path), rdA))
    print(report_line(os.path.basename(b_path), rdB))
    if rdA.get("n_used") and rdB.get("n_used"):
        d = rdB["rd_r1_mean"] - rdA["rd_r1_mean"]
        pct = 100 * d / rdA["rd_r1_mean"]
        print(f"\n变化：{d:+.4f} ({pct:+.1f}%)  "
              f"{'改善' if d < 0 else '恶化' if d > 0 else '持平'}")
    amap = {c["key"]: c["ratio"] for c in A["comparison"]}
    print("\n逐项（ratio 的变化，负数=更接近真实）：")
    for c in B["comparison"]:
        k = c["key"]
        if k not in amap:
            continue
        ra, rb = amap[k], c["ratio"]
        if not (math.isfinite(ra) and math.isfinite(rb) and ra > 0 and rb > 0):
            continue
        flag = "✅" if abs(math.log2(rb)) < abs(math.log2(ra)) else "❌"
        print(f"  {flag} {k:<22} {ra:<8} → {rb:<8}  "
              f"|log2| {abs(math.log2(ra)):.3f} → {abs(math.log2(rb)):.3f}")


# ---------------------------------------------------------------- F：三轨

def _patches(h, w, patch, per=5):
    pts = [(h // 2 - patch // 2, w // 2 - patch // 2)]
    per = max(1, per - 1)
    for i in range(per):
        fy = 0.2 + 0.6 * ((i % 2) if per > 1 else 0)
        fx = 0.2 + 0.6 * ((i // 2) / max(1, (per - 1) // 2))
        pts.append((int(fy * h) - patch // 2, int(fx * w) - patch // 2))
    out = []
    for y0, x0 in pts:
        out.append((max(0, min(h - patch, y0)), max(0, min(w - patch, x0))))
    return out


def struct_feats(path, patch=96, lag_lo=8, lag_hi=64, thr=0.30, per=5):
    """结构轨：局部 1D 自相关的峰高 + 重复单元计数（08 §9.5 的可测形式）。

    为什么用 1D 轮廓：2D 自相关在 96×96 上噪声太大；建筑立面/栏杆/窗格的主周期
    在行均值或列均值上就能看到。峰高 > thr 视为"存在重复单元"。
    """
    a = AG.load_rgb(path)
    g = AG.to_luma(a).astype(np.float32)
    h, w = g.shape
    peaks, counts = [], []
    for y0, x0 in _patches(h, w, patch, per):
        p = g[y0:y0 + patch, x0:x0 + patch]
        p = p - p.mean()
        if p.std() < 1e-6:
            continue
        for axis in (0, 1):
            prof = p.mean(axis=axis)
            prof = prof - prof.mean()
            if (prof ** 2).sum() < 1e-9:
                continue
            ac = np.correlate(prof, prof, "full")[len(prof) - 1:]
            if ac[0] <= 1e-9:
                continue
            ac = ac / ac[0]
            seg = ac[lag_lo:lag_hi]
            if not len(seg):
                continue
            peaks.append(float(seg.max()))
            counts.append(int((seg > thr).sum()))
    return {
        "acf_peak": round(float(np.mean(peaks)), 4) if peaks else None,
        "repeat_units": round(float(np.mean(counts)), 2) if counts else None,
        "n_patch": len(peaks),
    }


def depth_feats(beauty_path, zdir):
    """深度轨：台阶数 / AO 覆盖 / 接触强度 / tile-depth 相关。"""
    dn = np.asarray(Image.open(os.path.join(zdir, "depth.png")).convert("L"),
                    dtype=np.float32) / 255.0
    ao = np.asarray(Image.open(os.path.join(zdir, "ao.png")).convert("L"),
                    dtype=np.float32) / 255.0
    ct = np.asarray(Image.open(os.path.join(zdir, "contact.png")).convert("L"),
                    dtype=np.float32) / 255.0
    hist, _ = np.histogram(dn, bins=32, range=(0.0, 1.0))
    steps = len(_peaks(hist, rel=0.05))
    g = AG.to_luma(AG.load_rgb(beauty_path)).astype(np.float32)
    # tile-depth 相关：16px 块的局部对比度 vs 该块的平均深度（空气透视 → 负相关）
    bh, bw = g.shape[0] // 16 * 16, g.shape[1] // 16 * 16
    gb = g[:bh, :bw].reshape(bh // 16, 16, bw // 16, 16).std(axis=(1, 3)).ravel()
    db = dn[:bh, :bw].reshape(bh // 16, 16, bw // 16, 16).mean(axis=(1, 3)).ravel()
    corr = None
    if gb.std() > 1e-6 and db.std() > 1e-6:
        corr = float(np.corrcoef(gb, db)[0, 1])
    return {
        "depth_steps": int(steps),
        "ao_cover": round(float((1 - ao).mean()), 4),
        "contact_strength": round(float((1 - ct).mean()), 4),
        "tile_depth_corr": round(corr, 4) if corr is not None else None,
    }


def _peaks(hist, rel=0.15):
    m = hist.max()
    if m <= 0:
        return []
    on = hist > rel * m
    out = []
    for i in range(1, len(hist) - 1):
        if on[i] and hist[i] >= hist[i - 1] and hist[i] >= hist[i + 1]:
            out.append(i)
    return out


def _zdir_for(zdirs, syn_path):
    stem = os.path.basename(syn_path).replace("_beauty.png", "").replace(".png", "")
    if stem in ("scene", ""):
        stem = os.path.basename(os.path.dirname(os.path.abspath(syn_path)))
    for d in zdirs:
        if os.path.basename(d.rstrip("/")) == stem:
            return d
    return None


def three_track(syn, real, zdirs, baseline=None, baseline_zdirs=None, ref_rd=0.57):
    S = [AG.measure(p) for p in syn]
    R = [AG.measure(p) for p in real]
    cmp = build_comparison(S, R)
    rd = rel_dist(cmp)

    sf = [struct_feats(p) for p in syn]
    rf = [struct_feats(p) for p in real]

    def avg(rows, k):
        v = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
        return float(np.mean(v)) if v else None

    st_s = {k: avg(sf, k) for k in ("acf_peak", "repeat_units")}
    st_r = {k: avg(rf, k) for k in ("acf_peak", "repeat_units")}
    units_ratio = (st_s["repeat_units"] / st_r["repeat_units"]
                   if st_s["repeat_units"] and st_r["repeat_units"] else None)
    peak_ratio = (st_s["acf_peak"] / st_r["acf_peak"]
                  if st_s["acf_peak"] and st_r["acf_peak"] else None)

    df = []
    for p in syn:
        z = _zdir_for(zdirs or [], p)
        if z and os.path.exists(os.path.join(z, "depth.png")):
            df.append(depth_feats(p, z))
    dp = {k: avg(df, k) for k in ("depth_steps", "ao_cover", "contact_strength",
                                  "tile_depth_corr")}

    pass_stat = rd.get("rd_r1_mean") is not None and rd["rd_r1_mean"] <= ref_rd * 1.02
    pass_struct = (units_ratio is not None and 0.5 <= units_ratio <= 2.0
                   and peak_ratio is not None and 0.7 <= peak_ratio <= 1.4)
    pass_depth = (dp["depth_steps"] is not None and dp["depth_steps"] >= 2
                  and dp["ao_cover"] is not None and 0.03 <= dp["ao_cover"] <= 0.50
                  and (dp["tile_depth_corr"] is None or dp["tile_depth_corr"] <= -0.02))

    print("=== F 三轨验收 ===")
    cmp_txt = ""
    if baseline is not None:
        d = rd["rd_r1_mean"] - baseline
        cmp_txt = (f"｜baseline {baseline:.4f} → {rd['rd_r1_mean']:.4f} "
                   f"({100*d/baseline:+.1f}%)")
    print(f"[统计轨] rd_r1={rd.get('rd_r1_mean')} (ref {ref_rd}, 阈值 ≤{ref_rd*1.02:.4f})"
          f"{cmp_txt} → {'PASS' if pass_stat else 'FAIL'}")
    print(f"[结构轨] acf_peak 合成/真实={peak_ratio and round(peak_ratio,3)} "
          f"(0.7–1.4)｜repeat_units={units_ratio and round(units_ratio,3)} (0.5–2.0) "
          f"→ {'PASS' if pass_struct else 'FAIL'}")
    print(f"           合成 units={st_s['repeat_units']} peak={st_s['acf_peak']}｜"
          f"真实 units={st_r['repeat_units']} peak={st_r['acf_peak']}")
    print(f"[深度轨] steps={dp['depth_steps']} (≥2)｜ao_cover={dp['ao_cover']} (0.03–0.50)｜"
          f"tile-depth corr={dp['tile_depth_corr']} (≤-0.02) → {'PASS' if pass_depth else 'FAIL'}")
    if df:
        print(f"           逐景 steps={[d['depth_steps'] for d in df]} "
              f"ao={[d['ao_cover'] for d in df]} corr={[d['tile_depth_corr'] for d in df]}")
    print(f"\n结论：{'三轨全 PASS' if (pass_stat and pass_struct and pass_depth) else '有不通过项（见上）'}")
    return {"stat": {"rd": rd, "pass": pass_stat},
            "struct": {"syn": st_s, "real": st_r, "units_ratio": units_ratio,
                       "peak_ratio": peak_ratio, "pass": pass_struct},
            "depth": {"agg": dp, "per_scene": df, "pass": pass_depth}}


# ---------------------------------------------------------------- 语义轨（F 的缺口）

SEM_RULES = {
    # kind: (min_luma_std, max_luma_std, max_edge_p90, max_hf_ratio, note)
    "trees":    (0.015, None, 0.14, None, "叶簇要有多层次，不能是纯色黑球"),
    "foliage":  (0.010, None, 0.16, None, "单个叶簇同理"),
    "ground":   (0.008, None, None, None, "地面要有颗粒/接缝，不能整块纯色"),
    "sky":      (None, None, 0.06, 0.06, "天空要平滑，不能有硬边/高频"),
    "bldg":     (0.004, None, None, 0.55, "幕墙高频过高 = 雪花/噪点"),
}


def _kind_mask(ids, meta, kind):
    kinds = meta.get("id_kinds", {})
    gids = [int(k) for k, v in kinds.items() if v == kind]
    if not gids:
        return None
    return np.isin(ids, gids)


def semantic_feats(path, zdir):
    """按 zdir 的物体类别掩码，逐类算「物理合理区间」。

    这不是语义分割，是一次**廉价的物体级体检**：能抓住"树是纯黑球""天空有硬边"
    "幕墙像雪花"这三类三轨/盲测都看不见的错误（`11` §8.4 的判据缺口）。
    """
    meta = json.load(open(os.path.join(zdir, "zmeta.json"), encoding="utf-8"))
    ids = np.load(os.path.join(zdir, "ids.npy"))
    a = AG.load_rgb(path)
    g = AG.to_luma(a)
    if g.shape != ids.shape:
        from PIL import Image as _I
        g = np.asarray(_I.fromarray((g * 255).astype(np.uint8)).resize(
            (ids.shape[1], ids.shape[0])), dtype=np.float32) / 255.0
        a = np.asarray(_I.open(path).convert("RGB").resize(
            (ids.shape[1], ids.shape[0])), dtype=np.float32) / 255.0
    gy, gx = np.gradient(g)
    grad = np.hypot(gx, gy)
    out = {}
    for kind in SEM_RULES:
        m = _kind_mask(ids, meta, kind)
        if m is None or m.sum() < 200:
            continue
        # 腐蚀 9px：zdir 的 id 掩码是 **bbox**，不腐蚀会把背景算进物体内部，
        # 于是"纯色黑球"也会显示出方差（旧版树 std=0.037 的假阳性就是这么来的）。
        m = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).filter(
            ImageFilter.MinFilter(9)), dtype=np.uint8) > 127
        if m.sum() < 120:
            continue
        vals = g[m]
        gv = grad[m]
        # 高频比：2px 尺度的残差能量 / 总能量
        small = Image.fromarray((g * 255).astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(1.6))
        low = np.asarray(small, dtype=np.float32) / 255.0
        hf = np.abs(g - low)[m]
        out[kind] = {
            "n": int(m.sum()),
            "luma_mean": round(float(vals.mean()), 4),
            "luma_std": round(float(vals.std()), 4),
            "edge_p90": round(float(np.percentile(gv, 90)), 4),
            "hf_ratio": round(float(hf.mean() / max(vals.mean(), 1e-6)), 4),
        }
    return out


def semantic_track(syn, zdirs):
    rows = []
    for p in syn:
        z = _zdir_for(zdirs or [], p)
        if not z:
            continue
        rows.append({"file": os.path.basename(p), **semantic_feats(p, z)})
    if not rows:
        return None
    fails = []
    for r in rows:
        for kind, (mn_std, mx_std, mx_edge, mx_hf, note) in SEM_RULES.items():
            f = r.get(kind)
            if not f:
                continue
            if mn_std is not None and f["luma_std"] < mn_std:
                fails.append(f"{r['file']} {kind} luma_std={f['luma_std']}<{mn_std}（{note}）")
            if mx_edge is not None and f["edge_p90"] > mx_edge:
                fails.append(f"{r['file']} {kind} edge_p90={f['edge_p90']}>{mx_edge}（{note}）")
            if mx_hf is not None and f["hf_ratio"] > mx_hf:
                fails.append(f"{r['file']} {kind} hf_ratio={f['hf_ratio']}>{mx_hf}（{note}）")
    print("=== 语义轨（物体级体检）===")
    for r in rows:
        parts = []
        for kind, f in r.items():
            if kind == "file":
                continue
            parts.append(f"{kind}: std={f['luma_std']} edge={f['edge_p90']} hf={f['hf_ratio']}")
        print(f"  {r['file']}: " + "｜".join(parts))
    if fails:
        print(f"  → FAIL {len(fails)} 项：")
        for x in fails[:12]:
            print("    ✗ " + x)
    else:
        print("  → PASS（各物体的亮度方差/边缘/高频都在合理区间）")
    return {"rows": rows, "fails": fails, "pass": not fails}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report")
    ap.add_argument("--include", help="正则，只保留匹配的合成图后重算")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument("--syn", nargs="+")
    ap.add_argument("--real", nargs="+")
    ap.add_argument("--out")
    ap.add_argument("--three-track", action="store_true")
    ap.add_argument("--zdirs", nargs="+")
    ap.add_argument("--ref-rd", type=float, default=0.57,
                    help="统计轨参考 rd（默认 = J 配置的 0.5600 上浮 2%）")
    ap.add_argument("--baseline", help="baseline 的 gap_report.json（取其 rd 作为参照并打印 Δ）")
    ap.add_argument("--semantic", action="store_true", help="语义轨：按物体类别做物理性体检")
    args = ap.parse_args()

    if args.semantic:
        if not (args.syn and args.zdirs):
            sys.exit("--semantic 需要 --syn 与 --zdirs")
        syn = [p for pat in args.syn for p in sorted(glob.glob(pat))]
        zdirs = [d for pat in args.zdirs for d in sorted(glob.glob(pat))]
        semantic_track(syn, zdirs)
        return 0

    if args.three_track:
        if not (args.syn and args.real):
            sys.exit("--three-track 需要 --syn 与 --real")
        syn = [p for pat in args.syn for p in sorted(glob.glob(pat))]
        real = [p for pat in args.real for p in sorted(glob.glob(pat))]
        zdirs = [d for pat in (args.zdirs or []) for d in sorted(glob.glob(pat))]
        base_rd = None
        if args.baseline:
            base_rd = rel_dist(load_report(args.baseline)["comparison"])["rd_r1_mean"]
        three_track(syn, real, zdirs, baseline=base_rd,
                    ref_rd=base_rd if base_rd else args.ref_rd)
        return 0

    if args.compare:
        return compare(*args.compare)

    if args.syn and args.real:
        S = [AG.measure(p) for pat in args.syn for p in glob.glob(pat)]
        R = [AG.measure(p) for pat in args.real for p in glob.glob(pat)]
        if not S or not R:
            sys.exit("输入为空")
        rep = {"synthetic": S, "real": R, "comparison": build_comparison(S, R)}
        return emit(rep, args.out)

    if args.report:
        rep = load_report(args.report)
        if args.include:
            # 子集只用于观察，默认不覆盖全量报告（要落盘必须显式 --out）
            rep = subset(rep, args.include)
            return emit(rep, args.out, tag=f"subset:{args.include}")
        return emit(rep, args.out or args.report, tag=os.path.basename(args.report))

    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
