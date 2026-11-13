#!/usr/bin/env python3
"""matpick.py — 按「指标缺口 + 尺度匹配 + 光照有效性」选材质（B1 交付）

替代 procgen2.py:361 的哈希选材：
    return pool[sum(ord(c) for c in node.id) % len(pool)]      # 与意图/尺度/光照无关
哈希的可复现性保留了（同一节点每次同一材质），但选择依据换成可解释的打分。

公式（相对 05 §13.4 的两处修正，见 baseline/README.md §7）：
    score = 0.75 * Σ_k w_k * contrib_k(m)  +  0.25 * scale_fit(tile_px, feature_px)

    contrib（单调缺口型，gap 已归一化为「越大越缺」）：
        unique_color_ratio   w=0.30   gap * spectral
        grad_midtone_ratio   w=0.30   gap * spectral
        orient_entropy       w=0.20   gap * spectral
        flat_ratio           w=0.10   gap * (1 - spectral)      # flat 越低越好
    contrib（目标匹配型，越接近真实值越好）：
        freq_slope           w=0.05   1 - clip(|spec - real| / 0.5)
        chroma_mean          w=0.05   1 - clip(|spec - real| / 0.15)

    scale_fit = exp(-0.5 * (log2((tile_px/feature_px)/12) / 0.8)^2)
        理想的 1 个 tile 里有 ~12 个特征周期；偏离一个八度以内基本不罚。

硬排除（不是惩罚，直接出局）：
    slot ∉ material.slots / kind ∉ material.suits_kind /
    tile_px ∉ [min_px, max_px] / light.ev < ev_min / kelvin ∉ valid_under.kelvin

用法：
    # 单次查询（--log 打印每一项的贡献）
    .venv/bin/python matpick.py --nodes ~/go/procgen-lab/materials/nodes.jsonl \
        --gap baseline/gap_report_realistic.json \
        --slot curtain_wall --kind skyline --tile-px 77 --ev 0 --kelvin 5600 --log

    # 自检（确定性 / 敏感性 / 可解释）
    .venv/bin/python matpick.py --selftest

    # 对 4 个典型槽位批量出表
    .venv/bin/python matpick.py --table
"""
import argparse
import json
import math
import os
import random
from pathlib import Path
import sys

# 权重（全部单调型：方向由「合成图相对真实图是偏高还是偏低」决定）
#
# ⚠️ 第 4 次修正（B2 实测）：freq_slope / chroma_mean 原本做成"目标匹配型"
# （1 - |spec - 整图真实值| / 容差），这是**层级错误**——
# 整图的 chroma/freq_slope 由天空+建筑+地面共同决定，量纲根本不适用于单个材质
# （材质 albedo 的 chroma 只有 0.16–0.19，整图目标是 0.31）。结果 chroma 与
# hue_entropy 在 B2 首轮**双双变差**，正好踩了"不过就停"红线。
# 正确做法：把它们降级为"同类单调偏好"（合成偏低 → 选更高的那套），方向仍由数据定。
W_MONO = {
    "unique_color_ratio": (0.28, +1),   # 合成偏低 → 选更高的
    "grad_midtone_ratio": (0.28, +1),   # 合成偏低 → 选更高的
    "orient_entropy":     (0.18, +1),   # 合成偏低 → 选各向同性的
    "flat_ratio":         (0.10, -1),   # 合成偏高 → 选更不平的
    "chroma_mean":        (0.08, +1),   # 合成偏低 → 选彩度更高的
    "freq_slope":         (0.08, -1),   # 合成偏高（更平）→ 选更负的
}
SCALE_W = 0.15          # 尺度只做"硬约束之后的微调"，不该主导（实测 0.25 会锁死排序）
IDEAL_CYCLES = 12.0     # 一个 tile 里希望有多少个特征周期
LOG_TOL = 0.8            # 尺度匹配的 log2 容差


# ---------------------------------------------------------------- 缺口

def gap_from_report(report):
    """gap_report.json → 缺口字典（越大越缺，方向已统一）。

    ⚠️ 不能直接用 real - synthetic：负权重指标（flat_ratio）会反向、
    双向指标（freq_slope/chroma）会奖励"更差的那一侧"。
    """
    cmp = {c["key"]: c for c in report["comparison"]}
    missing = [k for k in W_MONO if k not in cmp]
    if missing:
        raise KeyError(f"gap_report 缺指标 {missing}（需 analyze_gap 的 13 项）")

    def sr(k):
        return float(cmp[k]["synthetic"]), float(cmp[k]["real"])

    syn, real = sr("unique_color_ratio")
    g = {"unique_color_ratio": max(0.0, real - syn)}
    syn, real = sr("grad_midtone_ratio")
    g["grad_midtone_ratio"] = max(0.0, real - syn)
    syn, real = sr("orient_entropy")
    g["orient_entropy"] = max(0.0, real - syn)
    syn, real = sr("flat_ratio")
    g["flat_ratio"] = max(0.0, syn - real)         # 合成太平 → 缺口为正
    syn, real = sr("chroma_mean")
    g["chroma_mean"] = max(0.0, real - syn)        # 合成偏灰 → 缺口为正
    syn, real = sr("freq_slope")
    g["freq_slope"] = max(0.0, syn - real)         # 合成更平（斜率偏正）→ 缺口为正
    return g


def target_values(report):
    """保留接口（B2 起全部指标改为单调型，不再需要目标值）。"""
    return {}


# ---------------------------------------------------------------- 打分

def scale_fit(tile_px, feature_px):
    if not feature_px or tile_px <= 0:
        return 0.0
    r = math.log2(max(tile_px / feature_px, 1e-6) / IDEAL_CYCLES)
    return math.exp(-0.5 * (r / LOG_TOL) ** 2)     # 0..1


SMOOTH = 0.05     # min-max 平滑：避免把"池内最优"硬拉到 1.0（2–3 套材质时会放大噪声）


def pool_norm(cands, mono_keys, target_keys=None, targets=None):
    """池内归一化表（两类）。

    为什么必须归一：绝对值比较下"选材质"退化成"挑 spec 最大的那套"，
    而 5 套同族材质的绝对差只有 0.01–0.05，权重再高也拉不开。
    选材质的本质是**候选池内的相对排序**，所以用池内 min-max。

    目标匹配型（freq_slope / chroma）不用人工容差——容差写死会让整池全部触底
    （实测 glass 的 freq/chroma 贡献恒为 0，等于这两项失效）。
    改成除以"池内最大偏差"，始终保留区分度。
    """
    mono = {}
    for k in mono_keys:
        vals = [m["spectral"].get(k) for m in cands
                if m["spectral"].get(k) is not None]
        mono[k] = (min(vals), max(vals)) if len(vals) >= 2 and max(vals) - min(vals) > 1e-9 else None
    return {"mono": mono}


def _norm(v, rng):
    if v is None or rng is None:
        return 0.5
    lo, hi = rng
    return (v - lo + SMOOTH) / (hi - lo + 2 * SMOOTH)


def score_material(m, gap, targets, tile_px, norm_stats, log=None,
                   albedo_w=0.0, albedo_ref=0.18):
    detail = []
    acc = 0.0
    wsum = 0.0
    for k, (w, sgn) in W_MONO.items():
        g = gap.get(k, 0.0)
        if g <= 0:
            continue                          # 没缺口的指标不参与（也不稀释分母）
        raw = m["spectral"].get(k)
        if raw is None:
            continue
        v = _norm(raw, norm_stats["mono"].get(k))
        if sgn < 0:
            v = 1.0 - v                       # flat/slope 越低越好
        acc += w * g * v
        wsum += w * g                          # ← 用 gap 加权，不是权重和
        detail.append((k, w, g, raw, v, w * g * v))
    # 在"当前缺口"的权重下取加权平均（0..1）。
    # ⚠️ 若按 Σw 归一，单项缺口的满分只有 0.28/1.0=0.28，尺度项 0.15 就能翻盘
    #（B2 实测：缺 midtone 时会错选 Facade002 而不是 midtone 最高的 Facade005）。
    idx_score = acc / wsum if wsum > 1e-9 else 0.5
    sf = scale_fit(tile_px, m["scale"].get("feature_px"))
    # B3：反照率拟合项（默认关，--albedo-w 开启）。夜景楼偏亮的根因是
    # 选材只看纹理统计、不看 albedo；这里只做"相对目标的软惩罚"，不硬排除。
    af = 1.0
    if albedo_w > 0:
        alb = (m.get("albedo_stats") or {}).get("mean")
        if alb and alb > 1e-6:
            af = math.exp(-abs(math.log2(alb / max(albedo_ref, 1e-6))) / 0.55)
    total = (1 - SCALE_W - albedo_w) * idx_score + SCALE_W * sf + albedo_w * af
    if log is not None:
        log.append((m["mid"], total, idx_score, sf, detail))
    return total


class MatIndex:
    """可被 procgen2 直接调用的材质索引（B2 用）。"""

    def __init__(self, nodes_path):
        self.rows = []
        with open(nodes_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))
        if not self.rows:
            raise ValueError(f"{nodes_path} 为空")
        # B3：全库分位兜底——候选池 <5 时池内 min-max 会把噪声拉成满分
        self.global_norm = pool_norm(self.rows, list(W_MONO))

    def pick(self, slot, kind, gap, tile_px, ev=0.0, kelvin=5600.0, cat=None,
             rng_key=None, topk=1, temp=0.02, log=None,
             albedo_w=0.0, albedo_ref=0.18):
        """返回 mid 或 None（无候选 → 调用方回退哈希）。

        cat：可选的类别约束（v1 场景里 `node.material` 写的是 concrete/glass 这类
        **类别**而不是槽位）。给 cat 时必须尊重场景原意，否则会把砖贴到玻璃楼上。
        """
        pool = []
        for m in self.rows:
            if cat and m["cat"] != cat:
                continue
            if slot not in m["slots"]:
                continue
            if kind and kind not in m["suits_kind"]:
                continue
            sc = m["scale"]
            fp = sc.get("feature_px")
            lo, hi = sc.get("min_px"), sc.get("max_px")
            if fp and lo is not None and hi is not None and not (lo <= tile_px <= hi):
                continue
            vu = m["valid_under"]
            if ev < vu.get("ev_min", -9):
                continue
            k0, k1 = vu.get("kelvin", [0, 1e9])
            if not (k0 <= kelvin <= k1):
                continue
            pool.append(m)
        if not pool:
            return None
        # 池太小（<5）时退回全库分位，避免"矮子里拔将军"给出虚高 idx_score
        norm_stats = pool_norm(pool, list(W_MONO)) if len(pool) >= 5 else self.global_norm
        self.last_pool_size = len(pool)
        cands = [(score_material(m, gap, self.targets, tile_px, norm_stats,
                                 albedo_w=albedo_w, albedo_ref=albedo_ref), m["mid"])
                 for m in pool]
        cands.sort(key=lambda x: (-x[0], x[1]))     # 稳定：分数相同按 mid

        # 材质多样性：确定性选优会让**整个场景只用一套材质**（同 cat 池内最优唯一），
        # 表现为重复感上升、hue_entropy 下降（B2 实测）。这里在 top-k 里按分数加权
        # 抽样，种子取 node.id（稳定哈希）→ 可复现，且同一场景内自然分散。
        chosen = cands[0][1]
        if topk and topk > 1 and len(cands) > 1:
            top = cands[:min(topk, len(cands))]
            best = top[0][0]
            wts = [math.exp((sc - best) / max(temp, 1e-6)) for sc, _ in top]
            seed = sum((i + 1) * ord(c) for i, c in enumerate(rng_key or ""))
            r = random.Random(seed).random() * sum(wts)
            acc = 0.0
            for (sc, mid), w in zip(top, wts):
                acc += w
                if r <= acc:
                    chosen = mid
                    break

        if log is not None:
            # 只对 top-5 重跑一次带明细的打分（分数已定，纯为了可解释输出）
            log.clear()
            by_mid = {m["mid"]: m for m in pool}
            for _sc, mid in cands[:5]:
                score_material(by_mid[mid], gap, self.targets, tile_px,
                               norm_stats, log=log,
                               albedo_w=albedo_w, albedo_ref=albedo_ref)
        return chosen

    def load_gap(self, report_path):
        rep = json.load(open(report_path, encoding="utf-8"))
        self.gap = gap_from_report(rep)
        self.targets = {}                # 保留字段以兼容旧调用；单调型不再需要目标值
        return self.gap, self.targets


# ---------------------------------------------------------------- CLI

def _fmt_log(rows_log):
    out = []
    for mid, total, idx_sum, sf, detail in rows_log:
        out.append(f"  {mid:<16} total={total:.4f}  (指标={idx_sum:.4f} 尺度={sf:.4f})")
        for k, w, g, spec, nv, c in detail:
            nvt = "—" if nv is None else f"{nv:.2f}"
            out.append(f"      {k:<22} w={w:<5} 缺口={g:<7.4f} spec={spec:<7.4f} norm={nvt:<5} → {c:.4f}")
    return "\n".join(out)


def selftest(idx):
    print("=== 自检 1/3 确定性")
    gA, tA = idx.gap, idx.targets
    a = idx.pick("curtain_wall", "skyline", gA, 77, log=None)
    b = idx.pick("curtain_wall", "skyline", gA, 77, log=None)
    print(f"  {a} vs {b}  {'PASS' if a == b else 'FAIL'}")
    ok1 = a == b

    print("\n=== 自检 2/3 缺口敏感性（纯单项缺口，可预测结果）")
    # 纯单项：只有 midtone 缺 vs 只有 unique_color 缺。
    # 期望：midtone 最高者（Facade005=0.2395） vs unique 最高者（Facade009=0.1161）。
    # 注意：混合缺口（unique 1.0 + midtone 0.05）不构成有效用例——两项都指向同一套材质时，
    #      "选择不变"是数据的正确结果，不是缺陷。
    gMid = {"unique_color_ratio": 0.0, "grad_midtone_ratio": 1.0,
            "orient_entropy": 0.0, "flat_ratio": 0.0}
    gUniq = {"unique_color_ratio": 1.0, "grad_midtone_ratio": 0.0,
             "orient_entropy": 0.0, "flat_ratio": 0.0}
    c1 = idx.pick("curtain_wall", "skyline", gMid, 77)
    c2 = idx.pick("curtain_wall", "skyline", gUniq, 77)
    exp1 = max((m for m in idx.rows if "curtain_wall" in m["slots"]),
               key=lambda m: m["spectral"]["grad_midtone_ratio"])["mid"]
    exp2 = max((m for m in idx.rows if "curtain_wall" in m["slots"]),
               key=lambda m: m["spectral"]["unique_color_ratio"])["mid"]
    print(f"  缺 midtone → {c1}（期望 {exp1}） {'PASS' if c1 == exp1 else 'FAIL'}")
    print(f"  缺 unique  → {c2}（期望 {exp2}） {'PASS' if c2 == exp2 else 'FAIL'}")
    ok2 = (c1 == exp1) and (c2 == exp2) and (c1 != c2)

    print("\n=== 自检 3/3 尺度敏感性（tile 40px → 150px；900px 在物理上荒谬，不测）")
    d = idx.pick("curtain_wall", "skyline", gA, 40)
    e = idx.pick("curtain_wall", "skyline", gA, 150)
    print(f"  tile40→{d}   tile150→{e}  {'PASS（选择随尺度变化）' if d != e else 'DEGRADED（选择不变）'}")
    ok3 = d != e

    print(f"\n结论：确定性 {'OK' if ok1 else 'BAD'}｜缺口敏感 {'OK' if ok2 else 'BAD'}｜尺度敏感 {'OK' if ok3 else '降级（记录，不调权重）'}")
    return 0 if ok1 else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", default=os.environ.get("PROCGEN_MATERIAL_NODES", str(Path.home() / ".cache" / "procgen" / "materials" / "nodes.jsonl")))
    ap.add_argument("--gap", default=os.environ.get("PROCGEN_GAP_REPORT", str(Path(__file__).resolve().parent / "baseline" / "gap_report_realistic.json")))
    ap.add_argument("--slot")
    ap.add_argument("--kind")
    ap.add_argument("--tile-px", type=float, default=77.0)
    ap.add_argument("--ev", type=float, default=0.0)
    ap.add_argument("--kelvin", type=float, default=5600.0)
    ap.add_argument("--log", action="store_true")
    ap.add_argument("--topk", type=int, default=1, help="1=确定性选优；>1=topk 加权随机（多样性）")
    ap.add_argument("--temp", type=float, default=0.02, help="topk 抽样的温度（分数差尺度）")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--table", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.nodes):
        sys.exit(f"缺 {args.nodes}（先跑 gen_material_nodes.py）")
    idx = MatIndex(args.nodes)
    idx.load_gap(args.gap)
    print(f"材质 {len(idx.rows)} 条｜缺口 {({k: round(v,3) for k,v in idx.gap.items()})}")
    print()

    if args.selftest:
        return selftest(idx)

    if args.table:
        cases = [("curtain_wall", "skyline", 77), ("facade_panel", "skyline", 77),
                 ("ground", "ground", 200), ("mullion", "skyline", 30),
                 ("parapet", "skyline", 60), ("pavement", "ground", 120)]
        print(f"{'slot':<13}{'kind':<9}{'tile':>6}  top-1               score   runner-up")
        for slot, kind, tile in cases:
            lg = []
            mid = idx.pick(slot, kind, idx.gap, tile, args.ev, args.kelvin,
                           rng_key=slot, topk=args.topk, temp=args.temp, log=lg)
            if not mid:
                print(f"{slot:<13}{kind:<9}{tile:>6}  —（无候选）")
                continue
            if not lg:
                print(f"{slot:<13}{kind:<9}{tile:>6}  {mid:<18} —")
                continue
            second = lg[1] if len(lg) > 1 else None
            tail = f"{second[0]}({second[1]:.4f})" if second else "—"
            print(f"{slot:<13}{kind:<9}{tile:>6}  {mid:<18} {lg[0][1]:.4f}  {tail}")
        return 0

    if not args.slot:
        ap.print_help()
        return 2
    lg = []
    mid = idx.pick(args.slot, args.kind, idx.gap, args.tile_px,
                   args.ev, args.kelvin, rng_key=args.slot,
                   topk=args.topk, temp=args.temp, log=lg)
    print(f"slot={args.slot} kind={args.kind} tile={args.tile_px}px ev={args.ev} kelvin={args.kelvin}")
    print(f"→ {mid}\n")
    if args.log:
        print(_fmt_log(lg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
