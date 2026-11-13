#!/usr/bin/env python3
"""e3_holdout.py — E3：leave-one-out 验证"选材"不是只拟合验收参考

做法（5 折，每折留 1 张 ref 做 held-out）：
  1) 用其余 4 张 ref 与 A1 渲染算 gap → gap_fold.json；
  2) 同场景同种子渲染两组：hash（--no-matpick）与 matpick（--gap fold）；
  3) 用**留出的那 1 张 ref** 算 rd_r1，比较两组。

判据：held-out 上 matpick 仍优于 hash（5 折至少 4 折）。
用法：.venv/bin/python e3_holdout.py
"""
import glob
import json
import os
import subprocess
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_gap as AG          # noqa: E402
import verify as V                # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
MAT = os.environ.get("PROCGEN_MATERIALS", str(Path.home() / ".cache" / "procgen" / "materials"))
REF = os.environ.get("PROCGEN_REFERENCE_GLOB", str(Path.home() / "Pictures" / "procgen-reference" / "*.jpg"))
A1 = ["dusk_warm", "night_blue", "shenzhen", "style_day"]
OUT = os.environ.get("PROCGEN_E3_OUT", str(Path.home() / ".cache" / "procgen" / "e3"))
PY = sys.executable


def render(name, outdir, extra):
    os.makedirs(outdir, exist_ok=True)
    cmd = [PY, os.path.join(ROOT, "procgen2.py"),
           os.path.join(ROOT, "baseline/scenes", f"{name}.json"), outdir,
           "--gbuffer", "beauty", "--materials", MAT] + extra
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if r.returncode != 0:
        raise SystemExit(f"渲染失败 {name} {outdir}: {(r.stderr or '')[-300:]}")
    return os.path.join(outdir, "scene_beauty.png")


def main():
    refs = sorted(glob.glob(REF))
    R_all = [AG.measure(p) for p in refs]
    S_a1 = [AG.measure(os.path.join(ROOT, "baseline/render", f"{n}_beauty.png")) for n in A1]
    rows = []
    for k in range(len(refs)):
        R_in = [r for i, r in enumerate(R_all) if i != k]
        cmp = V.build_comparison(S_a1, R_in)
        fold_dir = os.path.join(OUT, f"f{k}")
        os.makedirs(fold_dir, exist_ok=True)
        gap_path = os.path.join(fold_dir, "gap_fold.json")
        json.dump({"comparison": cmp, "rel_dist": V.rel_dist(cmp),
                   "synthetic": S_a1, "real": R_in, "holdout": os.path.basename(refs[k])},
                  open(gap_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

        rd = {}
        for mode, extra in (("hash", ["--no-matpick"]),
                            ("mat", ["--gap", gap_path])):
            imgs = [render(n, os.path.join(fold_dir, mode, n), extra) for n in A1]
            S = [AG.measure(p) for p in imgs]
            rd[mode] = V.rel_dist(V.build_comparison(S, [R_all[k]]))["rd_r1_mean"]
        rows.append({"fold": k, "holdout": os.path.basename(refs[k]),
                     "hash": rd["hash"], "mat": rd["mat"],
                     "delta_pct": round(100 * (rd["mat"] - rd["hash"]) / rd["hash"], 2)})
        print(f"fold {k} holdout={os.path.basename(refs[k]):18s} "
              f"hash={rd['hash']:.4f} mat={rd['mat']:.4f} Δ={rows[-1]['delta_pct']:+.1f}%")

    wins = sum(1 for r in rows if r["delta_pct"] < 0)
    print(f"\nheld-out 上 matpick 更优：{wins}/{len(rows)} 折"
          f"（判据 ≥4/5）→ {'PASS' if wins >= len(rows) - 1 else 'FAIL'}")
    json.dump(rows, open(os.path.join(OUT, "e3_results.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("→", os.path.join(OUT, "e3_results.json"))
    return 0 if wins >= len(rows) - 1 else 1


if __name__ == "__main__":
    sys.exit(main())
