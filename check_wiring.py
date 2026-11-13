#!/usr/bin/env python3
"""check_wiring.py — 接线自检（13 §3.4，W1-W5）

W1 每个 active 的 producer 存在（文件 + 符号）
W2 每个 active 的 consumer 在 pipeline.py 有调用点（stage 名出现）
W3 每个 rejected 的 forbid_pattern 未被"非零启用"
W4 阶段产物链完整（下游引用了上游产物）
W5 最近 manifest 结构完整（stages / capabilities_used / outputs|skipped）

用法：
    check_wiring.py [--manifest PATH] [--json]
退出码：0 全过；1 有 FAIL。
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

PROCGEN = Path(__file__).resolve().parent
PIPE = PROCGEN / "pipeline.py"
CAPS = PROCGEN / "capabilities.json"
WIN_TMP = Path(os.environ.get("PROCGEN_TMP", str(Path.home() / ".cache" / "procgen" / "pipe")))

# W4：上游产物 → 必须被下游引用
FLOW = [
    ("build", ["scene_beauty.png", "scene_depth.png", "scene_lineart.png"]),
    ("zbuf", ["--zdir"]),
    ("raster", ["scene_meta.json"]),
    ("gen", ["r.png", "scene_beauty.png"]),
]


def fail(msg):
    return ("FAIL", msg)


def ok(msg):
    return ("PASS", msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    caps = json.loads(CAPS.read_text(encoding="utf-8"))["capabilities"]
    src = PIPE.read_text(encoding="utf-8") if PIPE.exists() else ""
    rows = []

    for c in caps:
        cid, st = c["id"], c["status"]
        prod, cons = c.get("producer") or "", c.get("consumer") or ""

        if st == "active":
            # W1
            fname, _, sym = prod.partition(":")
            p = PROCGEN / fname
            if not fname or not p.exists():
                rows.append(fail(f"W1 {cid}: producer 不存在 {prod}"))
            elif sym and sym not in p.read_text(encoding="utf-8", errors="ignore"):
                rows.append(fail(f"W1 {cid}: producer 符号缺失 {prod}"))
            else:
                rows.append(ok(f"W1 {cid}: producer {prod}"))
            # W2
            stage = cons.split(":")[-1] if cons.startswith("pipeline:") else None
            if not stage:
                rows.append(fail(f"W2 {cid}: consumer 未绑定 pipeline 阶段"))
            elif f'"{stage}"' not in src and f"'{stage}'" not in src:
                rows.append(fail(f"W2 {cid}: pipeline 无 stage {stage}"))
            else:
                rows.append(ok(f"W2 {cid}: consumer pipeline:{stage}"))

        elif st == "rejected":
            pat = c.get("forbid_pattern")
            if not pat:
                rows.append(fail(f"W3 {cid}: rejected 缺 forbid_pattern"))
                continue
            hits = list(re.finditer(pat, src))
            bad = []
            for m in hits:
                val = m.group(1) if m.groups() else "1"
                try:
                    if float(val) != 0.0:
                        bad.append(m.group(0))
                except ValueError:
                    bad.append(m.group(0))
            if bad:
                rows.append(fail(f"W3 {cid}: 被否决但仍启用 {bad[:3]}"))
            else:
                rows.append(ok(f"W3 {cid}: 未启用"))

        elif st == "todo":
            if not c.get("since"):
                rows.append(fail(f"W? {cid}: todo 缺 since"))
            else:
                rows.append(ok(f"W? {cid}: todo({c['since']}) 已登记"))

    # W4 产物链
    for stage, outs in FLOW:
        for o in outs:
            if o not in src:
                rows.append(fail(f"W4 {stage}: pipeline 未引用 {o}"))
                break
        else:
            rows.append(ok(f"W4 {stage}: 产物链完整"))

    # W5 manifest
    mp = Path(args.manifest) if args.manifest else None
    if mp is None:
        cands = sorted(WIN_TMP.glob("*/manifest.json"), key=lambda p: p.stat().st_mtime)
        mp = cands[-1] if cands else None
    if mp and mp.exists():
        m = json.loads(mp.read_text(encoding="utf-8"))
        if not m.get("stages"):
            rows.append(fail(f"W5 {mp}: stages 为空"))
        elif not m.get("capabilities_used"):
            rows.append(fail(f"W5 {mp}: capabilities_used 为空"))
        else:
            missing = [s["name"] for s in m["stages"]
                       if not s.get("outputs") and not s.get("skipped")]
            rows.append(fail(f"W5 {mp}: 阶段无产物 {missing}") if missing
                        else ok(f"W5 {mp.name}: 结构完整（{len(m['stages'])} 阶段）"))
    else:
        rows.append(ok("W5 无 manifest（跳过；pipeline run 后会检查）"))

    nf = sum(1 for r in rows if r[0] == "FAIL")
    if args.json:
        print(json.dumps({"pass": len(rows) - nf, "fail": nf,
                          "rows": [{"status": s, "msg": m} for s, m in rows]},
                         ensure_ascii=False, indent=2))
    else:
        for s, m in rows:
            print(f"{s}  {m}")
        print(f"\n{len(rows)-nf} PASS / {nf} FAIL")
    return 1 if nf else 0


if __name__ == "__main__":
    sys.exit(main())
