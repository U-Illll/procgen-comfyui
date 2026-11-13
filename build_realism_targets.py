#!/usr/bin/env python3
"""build_realism_targets.py — 从专业摄影参考集反推「真实度目标向量」

输入：assets/ref-pro/px-*.jpg + SOURCES-pexels.json（含 caption，用于分档）
输出：realism_targets.json —— 分「夜景 / 黄昏 / 日间」三档的 12 指标均值

用途：每个元素的 `realism_target` 从这里取；渲染后与它比，差异定位到五轴中的哪一轴。
"""
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from analyze_gap import KEYS, measure

D = os.environ.get("PROCGEN_REFERENCE_DIR", str(Path.home() / "Pictures" / "procgen-reference"))
OUT = os.environ.get("PROCGEN_REALISM_TARGETS", str(Path.home() / ".cache" / "procgen" / "realism_targets.json"))

NIGHT = ("night", "illuminated", "lit up", "dark", "lights", "light trail", "long exposure")
DUSK = ("dusk", "sunset", "sunrise", "twilight", "blue hour", "evening", "golden hour", "afterglow")


def bucket(caption):
    c = (caption or "").lower()
    if any(k in c for k in DUSK):
        return "dusk"
    if any(k in c for k in NIGHT):
        return "night"
    return "day"


def main():
    src = os.path.join(D, "SOURCES-pexels.json")
    meta = {}
    if os.path.exists(src):
        for r in json.load(open(src, encoding="utf-8")):
            meta[f"px-{r['id']}.jpg"] = r

    groups = {"night": [], "dusk": [], "day": []}
    for p in sorted(glob.glob(os.path.join(D, "px-*.jpg"))):
        m = meta.get(os.path.basename(p), {})
        b = bucket((m.get("caption") or "") + " " + " ".join(m.get("tags") or []))
        groups[b].append(p)

    result = {}
    print(f"{'档':<6}{'张数':>4}")
    for b, files in groups.items():
        print(f"{b:<6}{len(files):>4}")
        if not files:
            continue
        result[b] = {}
        for k, cn, _ in KEYS:
            result[b][k] = round(float(np.mean([measure(p)[k] for p in files])), 5)

    result["_all"] = {k: round(float(np.mean([measure(p)[k] for p in
                                              glob.glob(os.path.join(D, "px-*.jpg"))])), 5)
                      for k, _, _ in KEYS}
    result["_meta"] = {
        "source": "Pexels License, 24 photos, 2560x1707",
        "counts": {b: len(f) for b, f in groups.items()},
        "note": "由 build_realism_targets.py 生成；每个元素的 realism_target 从这里取",
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)

    print(f"\n{'指标':<22}{'夜景':>10}{'黄昏':>10}{'日间':>10}{'全体':>10}")
    print("-" * 64)
    for k, cn, _ in KEYS:
        v = [result.get(b, {}).get(k, float("nan")) for b in ("night", "dusk", "day")]
        print(f"{cn:<22}{v[0]:>10.4f}{v[1]:>10.4f}{v[2]:>10.4f}{result['_all'][k]:>10.4f}")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
