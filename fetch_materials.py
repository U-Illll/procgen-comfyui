#!/usr/bin/env python3
"""fetch_materials.py — 从 ambientCG 建本地扫描材质库（1K JPG，全通道）

API: https://ambientcg.com/api/v2/full_json
     ?type=Material&q=<kw>&limit=100&sort=Popular&include=downloadData

产出：
    <root>/<AssetId>/albedo.jpg          Color
    <root>/<AssetId>/normal.jpg          NormalGL（OpenGL 约定，+Y 向上）
    <root>/<AssetId>/roughness.jpg       Roughness
    <root>/<AssetId>/ao.jpg              AmbientOcclusion（有的材质没有）
    <root>/<AssetId>/displacement.jpg    Displacement（有的材质没有）
    <root>/<AssetId>/meta.json           {assetId, category, tags, maps, size, license}
    <root>/registry.json                 {category: [assetId, ...]}

用法：
    python fetch_materials.py --dry-run                      # 只看会下什么
    python fetch_materials.py --per-cat 6                    # 实下
    python fetch_materials.py --cats concrete,glass --per-cat 3

许可：ambientCG 全部 CC0（可商用、无需署名）。仍写进 meta 便于审计。
"""
import argparse
import io
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

API = "https://ambientcg.com/api/v2/full_json"
UA = "procgen-lab/0.1 (local research; +https://example.invalid)"

# 类别 -> 搜索关键词（按顺序尝试，命中即用）
CATS = {
    "concrete": ["concrete"],
    "brick": ["bricks", "brick"],
    "plaster": ["plaster", "stucco"],
    "metal": ["metal", "steel"],
    "glass": ["glass", "facade"],
    "wood": ["wood", "planks"],
    "stone": ["stone", "rock"],
    "asphalt": ["asphalt", "road"],
    "pavement": ["pavement", "paving", "road"],
    "tiles": ["tiles"],
    "roofing": ["roofing", "roof"],
    "fabric": ["fabric", "carpet"],
}

# zip 内文件名关键词 -> 标准名
MAPS = [
    ("color", "albedo"),
    ("normalgl", "normal"),
    ("normaldx", "normal_dx"),
    ("roughness", "roughness"),
    ("ambientocclusion", "ao"),
    ("displacement", "displacement"),
    ("metalness", "metalness"),
    ("emissive", "emissive"),
]


def api(**params):
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def search(kw, limit=100):
    d = api(type="Material", q=kw, limit=limit, sort="Popular", include="downloadData")
    return d.get("foundAssets", [])


def pick_1k(asset):
    """从 downloadFolders 里挑 1K-JPG 的 zip 链接。"""
    folders = asset.get("downloadFolders") or {}
    for f in folders.values():
        cats = (f.get("downloadFiletypeCategories") or {})
        for c in cats.values():
            for dl in c.get("downloads", []):
                if dl.get("attribute") == "1K-JPG":
                    return dl.get("downloadLink"), dl.get("size") or 0
    return None, 0


def fetch(url, timeout=180):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def unpack(data, dst):
    """解压并把通道文件名标准化，返回实际写出的通道集合。"""
    os.makedirs(dst, exist_ok=True)
    got = {}
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            name = os.path.basename(info.filename)
            low = name.lower()
            # 只保留 _1K-JPG_ 那一套（zip 里通常只有一套）
            if "_1k-jpg_" not in low and not low.endswith((".jpg", ".jpeg")):
                continue
            base = re.sub(r"^.*_1k-jpg_", "", low)
            base = os.path.splitext(base)[0]
            for key, std in MAPS:
                if base == key or base.replace("-", "") == key:
                    path = os.path.join(dst, std + ".jpg")
                    with open(path, "wb") as f:
                        f.write(z.read(info))
                    got[std] = os.path.getsize(path)
                    break
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("PROCGEN_MATERIALS", str(Path.home() / ".cache" / "procgen" / "materials")))
    ap.add_argument("--per-cat", type=int, default=6)
    ap.add_argument("--cats", default=",".join(CATS))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-mb", type=float, default=2000, help="总下载上限，防跑飞")
    args = ap.parse_args()

    cats = [c.strip() for c in args.cats.split(",") if c.strip()]
    os.makedirs(args.root, exist_ok=True)
    registry_path = os.path.join(args.root, "registry.json")
    registry = {}
    if os.path.exists(registry_path):
        with open(registry_path, encoding="utf-8") as f:
            registry = json.load(f)

    total_mb = 0.0
    plan = []
    seen = set()
    for cat in cats:
        if cat not in CATS:
            print(f"[skip] 未知类别 {cat}")
            continue
        kws = CATS[cat]
        found = []
        for kw in kws:
            found = search(kw)
            if found:
                break
        if not found:
            print(f"[{cat}] 无命中")
            continue
        # 只留名字里真的含本类关键词的（否则 stone 会搜到 Bricks）
        strict = [a for a in found if any(k in a["assetId"].lower() for k in kws)]
        pool = strict if len(strict) >= args.per_cat else found
        picked = []
        for a in pool:
            url, size = pick_1k(a)
            if not url:
                continue
            aid = a["assetId"]
            if aid in seen:
                continue
            seen.add(aid)
            picked.append((aid, url, size, a.get("tags", []), a.get("displayCategory")))
            if len(picked) >= args.per_cat:
                break
        print(f"[{cat}] 命中 {len(found)}（严格 {len(strict)}），取 {len(picked)}: {[p[0] for p in picked]}")
        for p in picked:
            plan.append((cat, *p))

    print(f"\n计划下载 {len(plan)} 套，合计 {sum(p[3] for p in plan)/1048576:.0f} MB "
          f"-> {args.root}")
    if args.dry_run:
        return

    done = {cat: [] for cat in cats}
    for cat, asset_id, url, size, tags, dcat in plan:
        if total_mb + size / 1048576 > args.max_mb:
            print(f"! 超过 --max-mb {args.max_mb}，停止")
            break
        dst = os.path.join(args.root, asset_id)
        if os.path.exists(os.path.join(dst, "albedo.jpg")):
            print(f"= {asset_id} 已存在，跳过")
            done[cat].append(asset_id)
            continue
        try:
            t0 = time.time()
            data = fetch(url)
            got = unpack(data, dst)
            if "albedo" not in got:
                print(f"! {asset_id}: 解压后没有 albedo，删除")
                continue
            with open(os.path.join(dst, "meta.json"), "w", encoding="utf-8") as f:
                json.dump({"assetId": asset_id, "category": cat, "displayCategory": dcat,
                           "tags": tags, "maps": got, "zipBytes": size,
                           "source": f"https://ambientcg.com/a/{asset_id}",
                           "license": "CC0"}, f, ensure_ascii=False, indent=1)
            total_mb += size / 1048576
            done[cat].append(asset_id)
            print(f"+ {asset_id:<16} {size/1048576:5.1f}MB {len(got)} 通道 "
                  f"{time.time()-t0:.1f}s -> {sorted(got)}")
        except Exception as e:
            print(f"! {asset_id} 失败: {type(e).__name__}: {e}")
        time.sleep(0.25)

    for cat, ids in done.items():
        if ids:
            registry[cat] = sorted(set(registry.get(cat, []) + ids))
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=1)
    n = sum(len(v) for v in registry.values())
    print(f"\nregistry: {len(registry)} 类 / {n} 套 -> {registry_path}")


if __name__ == "__main__":
    main()
