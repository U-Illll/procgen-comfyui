#!/usr/bin/env python3
"""make_golden.py — 生成/刷新视觉回归基线（13 §3.5）

对固定 3 场景跑一遍 pipeline（本地链，不依赖 ComfyUI），把 final.jpg 与 13 指标
存到 baseline/golden/。之后每次接线改动用 `verify --compare` 或 check_wiring 对照。

用法：
    .venv/bin/python make_golden.py            # 重建
    .venv/bin/python make_golden.py --check    # 只对比当前 vs golden
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROCGEN = Path(__file__).resolve().parent
VENV = Path(os.environ.get("PROCGEN_PYTHON", sys.executable))
GOLD = PROCGEN / "baseline" / "golden"
TMP = Path(os.environ.get("PROCGEN_TMP", str(Path.home() / ".cache" / "procgen" / "pipe")))
SCENES = ["night_blue", "dusk_warm", "style_day"]
KEYS = ["grad_kurtosis", "grad_midtone_ratio", "unique_color_ratio", "flat_ratio",
        "local_contrast_cv", "orient_entropy", "freq_slope", "noise_sigma"]


def measure(p):
    sys.path.insert(0, str(PROCGEN))
    import analyze_gap as ag
    m = ag.measure(str(p))
    return {k: m[k] for k in KEYS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    GOLD.mkdir(parents=True, exist_ok=True)
    ref = GOLD / "golden.json"
    old = json.loads(ref.read_text(encoding="utf-8")) if ref.exists() else {}

    out = {}
    for s in SCENES:
        scene = PROCGEN / "baseline" / "scenes" / f"{s}.json"
        if not scene.exists():
            print(f"skip {s}: 无场景")
            continue
        d = TMP / s
        if not args.check:
            subprocess.run([str(VENV), str(PROCGEN / "pipeline.py"), "run", str(scene),
                            "--out", str(d), "--no-gen"], check=True,
                           stdout=subprocess.DEVNULL)
            shutil.copy(d / "degrade" / "final.jpg", GOLD / f"{s}.jpg")
        jpg = GOLD / f"{s}.jpg"
        if not jpg.exists():
            continue
        m = measure(jpg)
        out[s] = m
        if args.check and s in old:
            print(f"== {s}")
            for k in KEYS:
                a, b = old[s].get(k), m[k]
                if a is None:
                    continue
                rel = (b - a) / (abs(a) + 1e-9)
                flag = "⚠" if abs(rel) > 0.10 and k != "freq_slope" else " "
                print(f"  {flag} {k:<22}{a:>10.4f} -> {b:>10.4f}  ({rel:+.1%})")

    if not args.check:
        ref.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"golden -> {ref}  ({len(out)} 场景)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
