#!/usr/bin/env python3
"""pick_pexels.py — 从 pexels-568k 元数据里按标签精确挑「城市/建筑」摄影

数据源：HF `CaptionEmporium/pexels-568k-internvl2` 的 train.jsonl.gz（254 MB，仅元数据）
        字段：id / tags / width / height / aspect_ratio / url / internvl2_caption
许可：Pexels License（免费商用，无需署名）

为什么不用图片相似度筛：低层特征（边缘/绿色/平滑度）根本区分不了
「建筑立面」和「树林/铁轨/海浪」——实测 24 张里只有 1 张是城市。
这里直接用人工打的 tags，命中率接近 100%。

用法：
    python pick_pexels.py --top 24
    python pick_pexels.py --top 24 --min-width 3000 --queries skyline,cityscape,skyscraper
"""
import argparse
import gzip
import json
import os
import re
import urllib.request
from pathlib import Path

META = os.environ.get("PEXELS_META", "/tmp/pexels_meta.jsonl.gz")
OUT = os.environ.get("PROCGEN_REFERENCE_OUT", str(Path.home() / "Pictures" / "procgen-reference"))
UA = "procgen-lab/0.1 (local research)"

# 标签权重：越"城市天际线"越高
WANT = {
    "skyline": 10, "cityscape": 9, "skyscraper": 8, "downtown": 7,
    "city": 4, "urban": 4, "architecture": 3, "building": 3,
    "high rise": 5, "highrise": 5, "tower": 4, "rooftop": 4,
    "street": 2, "traffic": 3, "bridge": 2, "night": 4,
}
STRONG = {"skyline", "cityscape", "skyscraper", "downtown", "high rise", "highrise"}
# 明显跑题的标签：命中这些且没有强城市标签 → 丢弃
BAD = {"beach", "ocean", "sea", "forest", "tree", "mountain", "animal", "dog", "cat",
       "flower", "food", "car", "portrait", "woman", "man", "baby", "wedding",
       "snow", "desert", "cattle", "bird", "insect", "underwater", "wave"}
NIGHT = ("night", "dusk", "sunset", "sunrise", "twilight", "blue hour", "evening", "golden hour")


def tags_of(row):
    t = (row.get("tags") or "").lower()
    if not t and row.get("title"):
        t = row["title"].lower()
    return set(x.strip() for x in re.split(r"[,;/]", t) if x.strip())


def score(row):
    tags = tags_of(row)
    s = 0.0
    hit_strong = False
    for kw, w in WANT.items():
        if kw in tags:
            s += w
            if kw in STRONG:
                hit_strong = True
    if not hit_strong:
        return 0.0, tags
    # caption 里的昼夜/氛围加分
    cap = ((row.get("internvl2_caption") or "") + " " +
           (row.get("cogvlm_caption") or "")).lower()
    for kw in NIGHT:
        if kw in cap:
            s += 3
            break
    if "reflection" in cap or "reflection" in tags:
        s += 1
    # 跑题惩罚
    if tags & BAD:
        s -= 6
    # 构图：横图优先（壁纸参考），且在 16:9~16:10 附近
    ar = row.get("aspect_ratio") or (row.get("width", 1) / max(1, row.get("height", 1)))
    if ar >= 1.75:
        s += 3
    elif ar >= 1.45:
        s += 2
    elif ar < 1.2:
        s -= 4
    return s, tags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=24)
    ap.add_argument("--min-width", type=int, default=3000)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cands = []
    n = 0
    with gzip.open(META, "rt", encoding="utf-8") as f:
        for line in f:
            n += 1
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("width", 0) < args.min_width:
                continue
            s, tags = score(row)
            if s <= 0:
                continue
            cands.append((s, row))
    print(f"扫描 {n} 条，命中 {len(cands)} 条候选")

    cands.sort(key=lambda x: -x[0])
    # 去重（同 slug 前缀视为同组，避免连拍刷屏）
    seen, picked = set(), []
    for s, row in cands:
        key = (row.get("slug") or "")[:18]
        if key in seen:
            continue
        seen.add(key)
        picked.append((s, row))
        if len(picked) >= args.top:
            break

    print(f"\n入选 {len(picked)} 张：")
    for s, r in picked:
        print(f"  {s:5.1f} id={r['id']:<8} {r['width']}x{r['height']:<5} "
              f"{(r.get('slug') or '')[:34]:<34} {','.join(sorted(tags_of(r)))[:52]}")

    ok = 0
    for s, r in picked:
        dst = os.path.join(args.out, f"px-{r['id']}.jpg")
        if os.path.exists(dst) and os.path.getsize(dst) > 50000:
            ok += 1
            continue
        url = r["url"].split("?")[0] + "?auto=compress&cs=tinysrgb&w=2560"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            if len(data) < 50000:
                print(f"  ! {r['id']} 太小 {len(data)}")
                continue
            with open(dst, "wb") as f:
                f.write(data)
            ok += 1
        except Exception as e:
            print(f"  ! {r['id']}: {type(e).__name__}: {e}")
    print(f"\n下载 {ok}/{len(picked)} -> {args.out}")

    with open(os.path.join(args.out, "SOURCES-pexels.json"), "w", encoding="utf-8") as f:
        json.dump([{"id": r["id"], "slug": r.get("slug"), "width": r["width"],
                    "height": r["height"], "tags": sorted(tags_of(r)),
                    "caption": (r.get("internvl2_caption") or "")[:300],
                    "license": "Pexels License", "score": round(s, 2)}
                   for s, r in picked], f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
