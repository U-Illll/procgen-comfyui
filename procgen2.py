#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""procgen2 —— 面向「光追 × 扩散」混合管线的程序化绘图内核

相对 v1 的六项升级（参考 penrose / p5.js / paper.js / LLM4SVG）：
  1. 场景图（Node/Scene）取代 if-elif 指令链；新增图层、z、parent、transform
  2. 全局光照模型（方位角/高程角/色温/大气透视）——所有图元自动受同一套光
  3. 约束布局（align / stack / attach），不写死坐标（Penrose 思路）
  4. 材质预设（concrete/glass/metal/foliage/water/fabric）+ SVG lighting 原语
  5. **G-buffer 多通道输出**：beauty / depth / normal / lineart / seg —— 接 ControlNet
  6. Blender 场景导出（bpy 脚本）+ 批量模式 + meta.json
  7. 向后兼容 v1 的 ops DSL（自动转换）

用法：
  python3 procgen2.py --demo out/                       # 全部通道
  python3 procgen2.py scene.json out/ --gbuffer all
  python3 procgen2.py scene.json --blender scene.py     # 导出 Blender 场景
  python3 procgen2.py --batch scenes/ out/ --gbuffer beauty,depth

材质（v2 / B2，2026-10-26）：
  --materials DIR      挂本地扫描材质库（registry.json + nodes.jsonl）
  --matpick            打分选材（默认开；缺 nodes.jsonl/gap 自动回退哈希）
  --no-matpick         回退 v1 哈希选材
  --mat-topk N         材质多样性（默认 3：top-k 按分数加权抽样，种子=node.id）
  --tile-grad 0|1      纹理梯度（默认 0=常数 tile；1=按 depth，实测统计收益≈0）
  --gap FILE           指标缺口报告（默认 baseline/gap_report_realistic.json）
  DEBUG_MAT=1          打印每个节点的实际选材（诊断"是否真的接上了"）

无第三方依赖（只有标准库 + 本地 Chrome）。
"""

import base64
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path

CHROME = os.environ.get("PROCGEN_CHROME", "chromium")
TMP = Path(os.environ.get("PROCGEN_TMP", str(Path.home() / ".cache" / "procgen2")))

MODES = ("beauty", "depth", "normal", "lineart", "seg")

# ---- 材质（L2）----
# 每类可有多套，按 node.id 稳定哈希选择（同一栋楼每次渲染用同一套材质）
MATERIAL_DEFAULT = {"bldg": "concrete", "ground": "asphalt", "landmark": "glass"}

# v2（B2）：v1 场景里 node.material 写的是**类别**（concrete/glass/…），
# 打分选型需要**槽位**。这张表只做推导，场景原意（cat）仍然优先生效。
CAT_TO_SLOT = {
    "glass": "curtain_wall", "metal": "mullion", "concrete": "facade_panel",
    "brick": "facade_panel", "plaster": "facade_panel", "wood": "facade_panel",
    "stone": "podium_stone", "asphalt": "ground", "pavement": "ground",
    "tiles": "pavement", "roofing": "roof_equip", "fabric": "awning",
}
# 少数图元的纹理尺度不该按 depth 推（地面要铺得大、远塔要能看见格），
# 给的是一格在屏幕上的像素数。其余图元走 depth 纹理梯度。
MAT_TILE_PX_BY_KIND = {"ground": 180.0, "landmark": 60.0}
# skyline 内部的子块 kind 是 "bldg"，但材质节点的 suits_kind 用的是图元名。
# 不做这个映射，bldg 会被 suits_kind 全量过滤 → **静默回退哈希**（B2 实测踩到）。
KIND_ALIAS = {"bldg": "skyline"}
# pattern 缩放：256px 纹理 → 画面 ±115px 一个 tile（建筑立面尺度）
MAT_TILE_SCALE = 0.30
# 光照叠加强度（alpha 混合；multiply 与 opacity 组合在 Chrome/SVG 上不可靠）
MAT_LIGHT_OP = 0.45
# 材质叠层强度（G2）：贴图原本 100% 覆盖，玻璃/石材的高对比斑点在暗楼上像"雪花"。
# 0.55 保留纹理梯度，又不让贴图的白点盖过立面主色。
MAT_ALBEDO_OP = 1.00
# 光照增益：lit_color("#ffffff") 返回的是“纯光照色”，直接用作亮度会偏亮（逆光应成剪影）
MAT_LIGHT_GAIN = 0.60
# 材质图的反照率归一化目标（真实 albedo 均值 ≈0.2；逆光场景再压暗）
MAT_ALBEDO_MEAN = 0.14

# ================================================================ 色彩与光照


def hex2rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def rgb2hex(r, g, b):
    return "#%02x%02x%02x" % tuple(int(max(0, min(255, v))) for v in (r, g, b))


def mix(c1, c2, t):
    a, b = hex2rgb(c1), hex2rgb(c2)
    t = max(0.0, min(1.0, t))
    return rgb2hex(*[a[i] + (b[i] - a[i]) * t for i in range(3)])


def shade(c, k):
    return rgb2hex(*[v * k for v in hex2rgb(c)])


def kelvin_to_rgb(k):
    """色温 → RGB（Tanner Helland 近似），用于给光源着色。"""
    t = k / 100.0
    if t <= 66:
        r = 255
        g = 99.4708025861 * math.log(t) - 161.1195681661 if t > 0 else 0
    else:
        r = 329.698727446 * ((t - 60) ** -0.1332047592)
        g = 288.1221695283 * ((t - 60) ** -0.0755148492)
    if t >= 66:
        b = 255
    elif t <= 19:
        b = 0
    else:
        b = 138.5177312231 * math.log(t - 10) - 305.0447927307
    return rgb2hex(max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))


DEFAULT_LIGHT = {
    "azimuth": 250.0,     # 0=画面上方(北)，顺时针角度
    "elevation": 12.0,    # 0=贴地平线，90=正上方
    "kelvin": 3200.0,     # 3200 暖 / 5600 日光 / 9000 冷
    "intensity": 1.0,
    "ambient": 0.28,
    "haze": 0.34,         # 大气透视强度
    "haze_color": "#cfe0ee",
}

# 图元的法线朝向（给 normal 通道 & 受光强弱）
NORMAL_HINT = {
    "ground": (0.0, 1.0, 0.0),      # 朝上
    "sky": (0.0, 0.0, 1.0),
    "vertical": (0.0, 0.0, 1.0),    # 朝相机
    "foliage": (0.0, 0.5, 0.87),
    "water": (0.0, 0.9, 0.44),
}


def depth_span(node, hint="vertical", span=0.045):
    """图元内部的归一化深度范围 (d_top, d_bottom)。

    depth 通道与 zbuf 必须共用这一个定义，否则 two 套深度语义会打架：
    同一栋楼在 depth.png 里是一个台阶、在 zbuf 里是另一个。
    纯常数深度会被 ControlNet 当成「一片平的纸卡」——每个物体只有一个灰度台阶。
    """
    d = max(0.0, min(1.0, node.depth))
    if hint == "ground":
        return d - span * 0.6, d + span * 1.8   # 地平线远 → 画面底近
    if hint == "vertical":
        # d 越大越近。立面顶部离相机更远 → 顶 d 小、底 d 大。
        # ⚠️ 原实现写反了（顶部 d 大 = 顶部更近），导致 zbuf 推出的楼体法线朝下（G=0.447）。
        return d - span * 0.6, d + span * 0.6
    return d - span, d + span                   # 叶团：顶部略远，底部略近


class Light:
    def __init__(self, **kw):
        self.__dict__.update(DEFAULT_LIGHT)
        self.__dict__.update({k: v for k, v in kw.items() if v is not None})
        self.color = kelvin_to_rgb(self.kelvin)

    def sample(self, x, y, depth=0.5, normal=(0.0, 0.0, 1.0)):
        """Lambert 受光 + 水平方向感 + 天空光。normal 用世界坐标（x右 y上 z朝镜头）。

        逆光时（elevation 低、立面朝镜头）N·L≈0 → 自动变成剪影，这正是日落的样子。
        """
        az = math.radians(self.azimuth)
        el = math.radians(max(1.0, self.elevation))
        L = (math.sin(az) * math.cos(el), math.sin(el), -math.cos(az) * math.cos(el))
        ndl = max(0.0, normal[0] * L[0] + normal[1] * L[1] + normal[2] * L[2])
        k = self.ambient + self.intensity * ndl
        # 水平方向感：靠光源一侧略亮（艺术化，非严格物理）
        lx = 0.5 + 0.5 * math.sin(az)
        k += (0.5 - abs(x - lx)) * 0.10
        # 天空光：朝上的面额外得到环境光
        k += max(0.0, normal[1]) * 0.12
        fog = ((1.0 - max(0.0, min(1.0, depth))) ** 1.15) * self.haze
        return k, self.color, fog


def lit_color(base, x, y, depth, light, hint="vertical"):
    """受光物体的颜色。保留剪影（k 可以很低），只做上下限保护防死黑/过曝。"""
    n = NORMAL_HINT.get(hint, NORMAL_HINT["vertical"])
    k, lc, fog = light.sample(x, y, depth, n)
    k = max(0.16, min(1.55, k))
    c = shade(base, k)
    c = mix(c, lc, max(0.04, min(0.34, (k - 0.25) * 0.42)))   # 光源染色：受光越强越染
    c = mix(c, light.haze_color, fog)                          # 大气透视
    return c


def ndl_guess(k):
    return max(0.0, min(1.0, (k - 0.16) / 1.39))


def emissive_color(base, depth, light):
    """自发光体（天空/太阳）：不吃方向光，只做轻微色温与雾。"""
    fog = ((1.0 - max(0.0, min(1.0, depth))) ** 1.15) * light.haze * 0.55
    c = mix(base, light.haze_color, fog)
    return mix(c, light.color, 0.08)


# ================================================================ 材质预设

MATERIALS = {
    # name: (基础色偏移, SVG filter 片段生成器 key)
    "concrete": dict(tex=0.55, rough=0.85, spec=0.06),
    "glass":    dict(tex=0.10, rough=0.15, spec=0.65),
    "metal":    dict(tex=0.20, rough=0.30, spec=0.80),
    "foliage":  dict(tex=0.95, rough=0.90, spec=0.03),
    "water":    dict(tex=0.35, rough=0.05, spec=0.95),
    "fabric":   dict(tex=0.70, rough=0.95, spec=0.02),
    "skin":     dict(tex=0.25, rough=0.55, spec=0.25),
}


# ================================================================ 场景图


class Node:
    _seq = 0

    def __init__(self, kind, params=None, nid=None, depth=0.5, material=None, parent=None):
        Node._seq += 1
        self.id = nid or f"{kind}_{Node._seq}"
        self.kind = kind
        self.params = dict(params or {})
        self.depth = depth
        self.material = material
        self.parent = parent
        self.bbox = None
        self._bbox_acc = None

    def grow(self, x0, y0, x1, y1):
        if self._bbox_acc is None:
            self._bbox_acc = [x0, y0, x1, y1]
        else:
            a = self._bbox_acc
            a[0] = min(a[0], x0); a[1] = min(a[1], y0)
            a[2] = max(a[2], x1); a[3] = max(a[3], y1)

    def close_bbox(self):
        self.bbox = tuple(self._bbox_acc) if self._bbox_acc else None


def jpeg_size(path):
    """纯标准库读 JPEG 尺寸（SOF 段）。用于材质 pattern 的原始像素尺寸。"""
    with open(path, "rb") as f:
        data = f.read()
    i = 2
    while i < len(data) - 9:
        if data[i] != 0xFF:
            i += 1
            continue
        m = data[i + 1]
        if m in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            h = (data[i + 5] << 8) | data[i + 6]
            w = (data[i + 7] << 8) | data[i + 8]
            return w, h
        seg = (data[i + 2] << 8) | data[i + 3]
        i += 2 + seg
    return 256, 256


class Scene:
    def __init__(self, w=1920, h=1080, seed=42, light=None, camera=None):
        self.w, self.h, self.seed = w, h, seed
        self.light = Light(**(light or {}))
        self.camera = dict(focal=35.0, sensor=36.0, fstop=2.8,
                           pitch=0.0, shift_y=0.0)
        self.camera.update(camera or {})
        self.nodes = []
        self.material_nodes = None      # matpick.MatIndex（v2 打分选型）
        self.material_gap = None        # 指标缺口（从 gap_report 载入）
        self.tile_grad = 0.0            # 纹理梯度强度：默认 0=常数 tile（v1 行为），1=按 depth
        self.mat_topk = 3               # 材质多样性：在 top-k 里按分数加权抽样（1=确定性选优）
        self.mat_temp = 0.02
        self.mat_res = "auto"           # 贴图分辨率：auto/256/512/1024（auto=tile_px 自动升档）
        self.mat_tile_scale = MAT_TILE_SCALE   # 一格纹理的屏幕尺度基准（256*0.30≈77px）
        self.albedo_w = 0.0             # B3：反照率拟合项权重（0=关；推荐试 0.15）
        self.albedo_ref = 0.18          # 目标反照率（白天 0.18；夜景用 0.10）

    def load_materials(self, root):
        """挂上本地扫描材质库（由 fetch_materials.py 建）。

        没有调用过这个方法时，行为与无材质版完全一致。
        """
        reg_path = os.path.join(root, "registry.json")
        if not os.path.exists(reg_path):
            print(f"[material] 缺 {reg_path}，材质关闭")
            return False
        with open(reg_path, encoding="utf-8") as f:
            reg = json.load(f)
        self.materials = {k: v for k, v in reg.items() if v}
        self.mat_root = root
        # v2：载入材质节点（打分选型的候选池）。缺文件时静默降级到哈希选型。
        nodes_path = os.path.join(root, "nodes.jsonl")
        if os.path.exists(nodes_path):
            try:
                import matpick
                self.material_nodes = matpick.MatIndex(nodes_path)
            except Exception as e:
                print(f"[material] nodes.jsonl 载入失败（{e}），回退哈希选型")
        return True

    def load_gap(self, path):
        """载入指标缺口（gap_report.json），启用打分选型。"""
        if self.material_nodes is None:
            return False
        import matpick
        rep = json.load(open(path, encoding="utf-8"))
        self.material_nodes.gap = matpick.gap_from_report(rep)
        self.material_nodes.targets = matpick.target_values(rep)
        self.material_gap = self.material_nodes.gap
        return True

    def add(self, kind, params=None, **kw):
        n = Node(kind, params, **kw)
        self.nodes.append(n)
        return n


def from_v1_ops(ops, w=1920, h=1080, seed=42, light=None):
    """把 v1 的 ops 指令序列转成场景图（保兼容）。"""
    sc = Scene(w, h, seed, light=light)
    depth_map = {"sky": 0.05, "sun": 0.1, "haze": 0.2, "skyline": 0.45,
                 "ground": 0.9, "trees": 0.95, "railing": 1.0,
                 "landmark": 0.6, "vignette": 1.0, "grain": 1.0, "blur": 1.0}
    for i, op in enumerate(ops):
        kind = op["type"]
        params = {k: v for k, v in op.items() if k != "type"}
        sc.add(kind, params, nid=f"{kind}{i}", depth=depth_map.get(kind, 0.5))
    return sc


# ================================================================ 约束布局


def apply_layout(scene, rules):
    """极简约束求解：align / stack / attach。只调整 params 里的显式坐标。"""
    by_id = {n.id: n for n in scene.nodes}
    for r in rules or []:
        op = r.get("op")
        ids = [i for i in r.get("ids", []) if i in by_id]
        if not ids:
            continue
        if op == "align":
            axis = r.get("axis", "x")
            mode = r.get("mode", "left")
            key = r.get("key", "x" if axis == "x" else "y")
            vals = [by_id[i].params.get(key) for i in ids if by_id[i].params.get(key) is not None]
            if not vals:
                continue
            v = {"left": min(vals), "top": min(vals), "right": max(vals),
                 "bottom": max(vals), "center": sum(vals) / len(vals)}.get(mode, min(vals))
            for i in ids:
                by_id[i].params[key] = v
        elif op == "stack":
            gap = r.get("gap", 0.0)
            axis = r.get("axis", "y")
            cursor = r.get("start", 0.0)
            for i in ids:
                by_id[i].params[axis] = cursor
                h = by_id[i].params.get("h", by_id[i].params.get("size", 0.1))
                cursor += h + gap
        elif op == "attach":
            src, dst = r.get("src"), r.get("dst")
            if src in by_id and dst in by_id:
                s, d = by_id[src], by_id[dst]
                for k in ("x", "y"):
                    if k in r:
                        s.params[k] = d.params.get(k, 0.5) + r[k]


# ================================================================ 编译器


BACKGROUND_KINDS = {"sky", "sun", "haze", "vignette", "grain", "blur"}

class Compiler:
    """把场景图编译成任意通道的 SVG。

    bucket=(lo,hi) 时只绘制 depth 落在该区间的元素（用于 2.5D 分层）。
    """

    def __init__(self, scene, mode="beauty", bucket=None):
        self.s = scene
        self.mode = mode
        self.bucket = bucket
        self.w, self.h = scene.w, scene.h
        self.light = scene.light
        self.rng = random.Random(scene.seed)
        self.defs = []
        self.body = []
        self._ids = 0
        self.seg_map = {}
        self._cur = None
        # ---- 材质 ----
        self.mat = getattr(scene, "materials", None)
        self.mat_root = getattr(scene, "mat_root", None)
        # v2：打分选型（有 nodes.jsonl + gap 才启用，否则 _material_of 走哈希）
        self.picker = getattr(scene, "material_nodes", None)
        self.gap = getattr(scene, "material_gap", None)
        self.light_ev = math.log2(max(self.light.intensity, 1e-3))
        self.mat_topk = getattr(scene, "mat_topk", 1)      # 材质多样性（1=确定性选优）
        self.mat_temp = getattr(scene, "mat_temp", 0.02)
        self._pat_cache = {}
        self.mat_assign = {}            # P1：node.id -> assetId（导出给逐像素光照）
        self.drawn = []          # 实际绘制过的图元（含 skyline 内部子块），供 zbuf 用

    def _rec(self, node, x0, y0, x1, y1, hint, shape="rect", geom=None):
        """记录一次实际绘制（zbuf / BOM / 统计都靠它）。

        scene.nodes 只有顶层节点；skyline 的子块 bldg、landmark 的楼层线都是
        在绘制过程中临时创建的，不记录就会漏掉（zbuf 会少一半几何）。

        shape/geom 是给 zbuf 用的**真实轮廓**：只按 bbox 光栅化会把圆形的
        树冠、梯形的塔身写成方块（实测 depth 里树变成"方块堆"）。
        """
        self.drawn.append((node, float(x0), float(y0), float(x1), float(y1), hint,
                           shape, geom))

    def _in_bucket(self, node):
        if self.bucket is None:
            return True
        return self.bucket[0] <= node.depth < self.bucket[1] or (
            self.bucket[1] >= 1.0 and node.depth >= self.bucket[0])

    # ---- 基础设施 ----
    def _id(self, prefix="p"):
        self._ids += 1
        return f"{prefix}{self._ids}"

    def _seg_color(self, node):
        if node.id not in self.seg_map:
            i = len(self.seg_map) + 1
            self.seg_map[node.id] = rgb2hex((i * 53) % 256, (i * 97) % 256, (i * 149) % 256)
        return self.seg_map[node.id]

    def style(self, node, x=0.5, y=0.5, hint="vertical", base="#888888"):
        """按当前通道给出填充色。所有图元的唯一取色入口。"""
        if self.mode == "seg":
            return self._seg_color(node)
        if self.mode == "depth":
            d = max(0.0, min(1.0, node.depth))
            g = int(15 + 240 * d)
            return rgb2hex(g, g, g)
        if self.mode == "normal":
            nx, ny, nz = NORMAL_HINT.get(hint, NORMAL_HINT["vertical"])
            return rgb2hex(128 + 127 * nx, 128 + 127 * ny, 128 + 127 * nz)
        if self.mode == "lineart":
            return "none"
        if node.params.get("silhouette"):
            # 剪影：不吃全局光照。逆光下的树/前景就该是纯暗色块，
            # 一旦让 kelvin 暖光染上去，就变成「棕色云团」。
            return base
        return lit_color(base, x, y, node.depth, self.light, hint)

    def stroke_attrs(self):
        return 'stroke="#000" stroke-width="2"' if self.mode == "lineart" else ""

    def _mat_tile_px(self, node):
        """一格纹理在屏幕上的像素数（纹理梯度）。

        物理依据：同一物理尺寸在近处投影更大 → tile_px 随 depth 增大。
        depth .3/.55/1.0 → 68/79/100 px（围绕旧常数 76.8px 摆动 ±30%）。
        地面/远塔用 MAT_TILE_PX_BY_KIND 覆盖（它们的投影尺度不该按楼体 depth 推）。
        """
        p = node.params or {}
        if p.get("tile_px"):
            return float(p["tile_px"])
        if node.kind in MAT_TILE_PX_BY_KIND:
            return MAT_TILE_PX_BY_KIND[node.kind]
        d = max(0.0, min(1.0, node.depth))
        base = 256.0 * getattr(self.s, "mat_tile_scale", MAT_TILE_SCALE)
        grad = (0.70 + 0.60 * d)                     # depth .3/.55/1.0 → .88/1.03/1.30
        g = max(0.0, min(1.0, getattr(self.s, "tile_grad", 1.0)))
        return base * (1.0 + (grad - 1.0) * g)

    def _material_of(self, node, slot=None):
        """给定节点 → assetId。v2 优先打分选型，失败/未启用则回退哈希。"""
        if not self.mat or not self.mat_root:
            return None
        cat = getattr(node, "material", None) or MATERIAL_DEFAULT.get(node.kind)
        if not cat:
            return None
        if self.picker is not None and self.gap is not None:
            aid = self.picker.pick(
                slot=slot or CAT_TO_SLOT.get(cat, cat),
                kind=KIND_ALIAS.get(node.kind, node.kind),
                gap=self.gap,
                tile_px=self._mat_tile_px(node),
                ev=self.light_ev,
                kelvin=self.light.kelvin,
                cat=cat,
                rng_key=node.id,
                topk=self.mat_topk,
                temp=self.mat_temp,
                albedo_w=getattr(self.s, "albedo_w", 0.0),
                albedo_ref=(getattr(self.s, "albedo_ref", 0.18)
                            if self.light_ev >= 0.6
                            else getattr(self.s, "albedo_ref", 0.18) * 0.55),
            )
            if aid:
                if os.environ.get("DEBUG_MAT"):
                    print(f"[mat] {node.kind:<8} cat={cat:<9} slot={slot or CAT_TO_SLOT.get(cat, cat):<13} "
                          f"tile={self._mat_tile_px(node):.0f}px -> {aid}")
                self.mat_assign[node.id] = aid
                return aid
        aid = self._material_of_hash(node)
        if aid:
            self.mat_assign[node.id] = aid
        return aid

    def _material_of_hash(self, node):
        """v1 行为（保留为回退路径）：按 node.id 稳定哈希选。"""
        if not self.mat or not self.mat_root:
            return None
        cat = getattr(node, "material", None) or MATERIAL_DEFAULT.get(node.kind)
        if not cat:
            return None
        pool = self.mat.get(cat)
        if not pool:
            # 允许直接写 assetId
            if os.path.isdir(os.path.join(self.mat_root, str(cat))):
                return str(cat)
            return None
        return pool[sum(ord(c) for c in node.id) % len(pool)]

    def _pattern_for(self, asset_id, scale=None):
        """把 albedo 缩略图以 data URI 内嵌成 pattern。

        v2：scale 参与缓存键——同一个 asset 在不同 depth 下 tile 大小不同，
        只按 asset_id 缓存会让第一个 depth 的缩放被全局复用（纹理梯度失效）。
        scale 量化到 3 位小数，限制同一 asset 的变体数（SVG 体积）。
        """
        sc = round(float(MAT_TILE_SCALE if scale is None else scale), 3)
        res = getattr(self.s, "mat_res", "auto")
        if res == "auto":
            # auto：只要求 ≥1:1 覆盖屏幕尺度。tile≤256px 时 256 已够（实测 PSNR>52dB）
            tile_px = 256.0 * sc
            res = "256" if tile_px <= 256 else ("512" if tile_px <= 512 else "1024")
        if res in ("orig", "1024"):
            res = "1024"          # 不直读 albedo.jpg：原图未做 gain 归一化（B3.1 双变量）
        key = (asset_id, sc, res)
        if key in self._pat_cache:
            return self._pat_cache[key]
        # 档位 → 文件名/边长；缺失时逐级回退（1024→512→256），保证亮度标定一致
        tiers = {"256": ("albedo_256.jpg", 256), "512": ("albedo_512.jpg", 512),
                 "1024": ("albedo_1024.jpg", 1024)}
        order = [res] + [r for r in ("1024", "512", "256") if r != res]
        path = tw = None
        for r in order:
            fn, px = tiers[r]
            p = os.path.join(self.mat_root, asset_id, fn)
            if os.path.exists(p):
                path, tw = p, px
                break
        if path is None:
            # 兜底：新下载的材质可能只有原图（未归一化，亮度会偏；总好过无材质）
            p = os.path.join(self.mat_root, asset_id, "albedo.jpg")
            if not os.path.exists(p):
                self._pat_cache[key] = None
                return None
            path = p
            tw, th = jpeg_size(path)
        else:
            th = tw
        # 单变量纪律：res 只改清晰度，不改 tile 的物理尺度 → 1024 图要按 256 基准反缩放
        sc_eff = sc * (256.0 / tw)
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        gid = self._id("mat")
        self.defs.append(
            f'<pattern id="{gid}" patternUnits="userSpaceOnUse" width="{tw}" height="{th}" '
            f'patternTransform="scale({sc_eff})">'
            f'<image href="data:image/jpeg;base64,{b64}" width="{tw}" height="{th}" '
            f'preserveAspectRatio="none"/></pattern>')
        self._pat_cache[key] = gid
        return gid

    def _depth_fill(self, node, hint):
        """depth 通道的层内梯度。

        纯常数深度的图元会被 ControlNet 当成「一片平的纸卡」——每个物体只有一个
        灰度台阶。真实世界没有这种面（地面向远处渐远、立面上下也有深度变化）。
        这里给每个图元加一个很小的线性梯度（span ≈5% 量级），让 depth 直方图
        从「几个尖峰」变成「连续带」，同时保留分层语义。
        """
        d0, d1 = depth_span(node, hint)

        def gray(v):
            g = int(15 + 240 * max(0.0, min(1.0, v)))
            return rgb2hex(g, g, g)

        return f"url(#{self.gradient(gray(d0), gray(d1), 0, 0, 0, 1)})"

    def emit_rect(self, node, x, y, w, h, base, hint="vertical", opacity=1.0):
        if not self._in_bucket(node):
            return
        cx, cy = (x + w / 2) / self.w, (y + h / 2) / self.h
        if self.mode == "depth":
            fill = self._depth_fill(node, hint)
        else:
            fill = self.style(node, cx, cy, hint, base)
        node.grow(x, y, x + w, y + h)
        self._rec(node, x, y, x + w, y + h, hint, "rect")
        st = self.stroke_attrs()
        if self.mode == "lineart":
            self.body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
                             f'fill="none" {st} opacity="{opacity}"/>')
        else:
            pid = None
            if self.mode == "beauty":
                aid = self._material_of(node)
                if aid:
                    pid = self._pattern_for(aid, self._mat_tile_px(node) / 256.0)
            if pid:
                # G2：材质层不再 100% 覆盖，所以必须先铺一层不透明 base，
                # 否则降低材质 opacity 会把后面的楼"透"出来（实测出现幽灵重叠）。
                self.body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
                                 f'fill="{fill}" opacity="{opacity}"/>')
                # 光照层必须是「纯光照」：base 用白色，否则会把材质二次着色（base²）
                light = shade(lit_color("#ffffff", cx, cy, node.depth, self.light, hint),
                              MAT_LIGHT_GAIN)
                self.body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
                                 f'fill="url(#{pid})" opacity="{opacity * MAT_ALBEDO_OP:.2f}"/>')
                self.body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
                                 f'fill="{light}" opacity="{MAT_LIGHT_OP * opacity:.2f}"/>')
            else:
                self.body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
                                 f'fill="{fill}" opacity="{opacity}"/>')

    def emit_poly(self, node, pts, base, hint="vertical", opacity=1.0):
        if not self._in_bucket(node):
            return
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        cx, cy = sum(xs) / len(xs) / self.w, sum(ys) / len(ys) / self.h
        if self.mode == "depth":
            fill = self._depth_fill(node, hint)
        else:
            fill = self.style(node, cx, cy, hint, base)
        node.grow(min(xs), min(ys), max(xs), max(ys))
        self._rec(node, min(xs), min(ys), max(xs), max(ys), hint, "poly", list(pts))
        d = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        if self.mode == "lineart":
            self.body.append(f'<polygon points="{d}" fill="none" {self.stroke_attrs()} opacity="{opacity}"/>')
        else:
            pid = None
            if self.mode == "beauty":
                aid = self._material_of(node)
                if aid:
                    pid = self._pattern_for(aid, self._mat_tile_px(node) / 256.0)
            if pid:
                light = shade(lit_color("#ffffff", cx, cy, node.depth, self.light, hint),
                              MAT_LIGHT_GAIN)
                self.body.append(f'<polygon points="{d}" fill="url(#{pid})" '
                                 f'opacity="{opacity * MAT_ALBEDO_OP:.2f}"/>')
                self.body.append(f'<polygon points="{d}" fill="{light}" '
                                 f'opacity="{MAT_LIGHT_OP * opacity:.2f}"/>')
            else:
                self.body.append(f'<polygon points="{d}" fill="{fill}" opacity="{opacity}"/>')

    def _blur_filter(self, px):
        """按 px 缓存一个 SVG 高斯模糊（叶簇软边用；P3）。"""
        if not hasattr(self, "_filters"):
            self._filters = {}
        key = f"{px:.2f}"
        if key not in self._filters:
            fid = self._id("blurf")
            pad = int(200 + px * 40)
            self.defs.append(
                f'<filter id="{fid}" x="-{pad}%" y="-{pad}%" width="{100 + 2*pad}%" '
                f'height="{100 + 2*pad}%"><feGaussianBlur stdDeviation="{px}"/></filter>')
            self._filters[key] = fid
        return self._filters[key]

    def emit_circle(self, node, cx, cy, r, base, hint="foliage", opacity=1.0, blur=0.85):
        if not self._in_bucket(node):
            return
        if self.mode == "depth":
            fill = self._depth_fill(node, hint)
        else:
            fill = self.style(node, cx / self.w, cy / self.h, hint, base)
        node.grow(cx - r, cy - r, cx + r, cy + r)
        self._rec(node, cx - r, cy - r, cx + r, cy + r, hint, "circle", (cx, cy, r))
        if self.mode == "lineart":
            self.body.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="none" '
                             f'{self.stroke_attrs()} opacity="{opacity}"/>')
        else:
            flt = f' filter="url(#{self._blur_filter(blur)})"' if blur > 0 else ""
            self.body.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}" '
                             f'opacity="{opacity}"{flt}/>')

    def gradient(self, c1, c2, x1=0, y1=0, x2=0, y2=1):
        gid = self._id("grad")
        self.defs.append(
            f'<linearGradient id="{gid}" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}">'
            f'<stop offset="0" stop-color="{c1}"/><stop offset="1" stop-color="{c2}"/>'
            f'</linearGradient>')
        return gid

    # ---- 图元 ----
    def sky(self, n):
        p = n.params
        horizon = p.get("horizon", 0.62)
        top = p.get("top", "#1e3a5f")
        bot = p.get("bottom", "#f2c98a")
        if self.mode == "lineart":
            n.grow(0, 0, self.w, horizon * self.h)
            self.body.append(f'<rect x="0" y="0" width="{self.w}" height="{horizon*self.h:.1f}" '
                             f'fill="none" stroke="#000" stroke-width="2"/>')
            return
        if self.mode in ("depth", "normal", "seg"):
            self.emit_rect(n, 0, 0, self.w, horizon * self.h + 2, bot,
                           hint="sky")
            return
        # beauty：天空是自发光体，只做色温/雾
        c_top = emissive_color(top, n.depth, self.light)
        c_bot = emissive_color(bot, n.depth, self.light)
        gid = self.gradient(c_top, c_bot, 0, 0, 0, horizon * 1.15)
        n.grow(0, 0, self.w, horizon * self.h)
        self.body.append(f'<rect x="0" y="0" width="{self.w}" height="{horizon*self.h+2:.1f}" '
                         f'fill="url(#{gid})"/>')

    def sun(self, n):
        p = n.params
        x, y = p.get("x", 0.72), p.get("y", 0.42)
        r = p.get("r", 0.06) * self.h
        color = p.get("color") or kelvin_to_rgb(max(1800, self.light.kelvin - 400))
        cx, cy = x * self.w, y * self.h
        if self.mode == "lineart":
            self.emit_circle(n, cx, cy, r * 0.55, color, hint="sky")
            return
        gid = self._id("glow")
        if self.mode == "beauty":
            c = emissive_color(color, n.depth, self.light)
            self.defs.append(
                f'<radialGradient id="{gid}"><stop offset="0" stop-color="{c}" stop-opacity="0.95"/>'
                f'<stop offset="0.35" stop-color="{c}" stop-opacity="0.45"/>'
                f'<stop offset="1" stop-color="{c}" stop-opacity="0"/></radialGradient>')
            self.body.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r*p.get("glow",2.6):.1f}" '
                             f'fill="url(#{gid})"/>')
        self.emit_circle(n, cx, cy, r * 0.55, color, hint="sky")

    def haze(self, n):
        p = n.params
        y = p.get("y", 0.55); hh = p.get("height", 0.18)
        color = p.get("color", self.light.haze_color)
        op = p.get("opacity", 0.35)
        if self.mode != "beauty":
            return
        gid = self._id("haze")
        self.defs.append(
            f'<linearGradient id="{gid}" x1="0" y1="0" x2="0" y2="1">'
            f'<stop offset="0" stop-color="{color}" stop-opacity="0"/>'
            f'<stop offset="0.5" stop-color="{color}" stop-opacity="{op}"/>'
            f'<stop offset="1" stop-color="{color}" stop-opacity="0"/></linearGradient>')
        n.grow(0, (y - hh / 2) * self.h, self.w, (y + hh / 2) * self.h)
        self.body.append(f'<rect x="0" y="{(y-hh/2)*self.h:.1f}" width="{self.w}" '
                         f'height="{hh*self.h:.1f}" fill="url(#{gid})"/>')

    def skyline(self, n):
        p = n.params
        horizon = p.get("horizon", 0.62)
        depth_layers = p.get("depth", 4)
        base = p.get("base_color", "#2a3b4d")
        window_p = p.get("window_p", 0.35)
        hmin, hmax = p.get("height_range", (0.08, 0.34))
        crowd = p.get("crowd", 1.0)
        rng = random.Random(p.get("seed", self.s.seed))
        # P2：默认不再把 glass 用于 bldg——glass 贴图自带高对比白斑，在 100px tile 下
        # 就是"幕墙雪花"。玻璃留给 landmark（远塔小、有窗格），bldg 只用低对比类别。
        mpool = p.get("material", ["concrete", "glass", "plaster", "brick"])
        if isinstance(mpool, str):
            mpool = [mpool]
        for layer in range(depth_layers):
            t = layer / max(1, depth_layers - 1)
            nd = max(0.02, min(1.0, n.depth * (0.55 + 0.45 * t)))
            color = mix(mix(base, self.light.haze_color, 0.55 * (1 - t)), base, 0.25 * t)
            hy = (horizon - 0.02 * (1 - t)) * self.h
            lo = hmin * self.h * (0.45 + 0.55 * t)
            hi = hmax * self.h * (0.45 + 0.55 * t)
            x = -60.0
            while x < self.w + 60:
                bw = rng.uniform(28, 120) * (1.15 - 0.3 * t) * crowd
                gap = rng.uniform(2, 22) * (1.4 - 0.4 * t)
                bh = rng.uniform(lo, hi)
                sub = Node("bldg", {}, nid=f"{n.id}_L{layer}_{int(x)}", depth=nd,
                           material=rng.choice(mpool) if mpool else None)
                # 退台：先画窄的上段，再画宽的下段（下段覆盖重叠区）
                if rng.random() < p.get("setback_p", 0.22):
                    sw = bw * rng.uniform(0.55, 0.85)
                    sh = bh * rng.uniform(0.08, 0.22)
                    self.emit_rect(sub, x + (bw - sw) / 2, hy - bh - sh, sw, sh + 2,
                                   color, hint="vertical")
                self.emit_rect(sub, x, hy - bh, bw, bh + 4, color, hint="vertical")
                # P1：体积感——右侧暗带（背光面）+ 顶部亮边（天光）。
                # 不加这两条，每栋楼就是一张正面平涂的纸片（G 阶段目视的主要破绽）。
                pass  # P1 已回退：实测 rd −2.4% 但盲测 AUC +0.11（反向）
                if rng.random() < 0.25:
                    nw = bw * rng.uniform(0.3, 0.55)
                    nh = bh * rng.uniform(0.06, 0.16)
                    self.emit_rect(sub, x + (bw - nw) / 2, hy - bh - nh, nw, nh + 2, color)
                # 屋顶设备/女儿墙：数量与位置随机（真实天台从不空）
                if rng.random() < p.get("roof_p", 0.55):
                    for _k in range(rng.randint(1, 3)):
                        rw = bw * rng.uniform(0.08, 0.30)
                        rh = bh * rng.uniform(0.010, 0.045)
                        rx = x + rng.uniform(0.04, 0.96) * max(0.0, bw - rw)
                        self.emit_rect(sub, rx, hy - bh - rh, rw, rh + 1, color,
                                       hint="vertical", opacity=0.97)
                # 幕墙竖向分格：细节必须有「物理宽度」（1px 细线是硬边，会拉高峰度）
                if bw > 40 and rng.random() < p.get("mullion_p", 0.30):
                    for _k in range(rng.randint(2, 4)):
                        mx = x + bw * rng.uniform(0.06, 0.94)
                        self.emit_rect(sub, mx, hy - bh, max(2.0, bw * 0.009), bh,
                                       shade(color, 1.10), hint="vertical", opacity=0.26)
                if t > 0.25 and window_p > 0 and rng.random() < window_p and self.mode == "beauty":
                    self._windows(sub, x, hy - bh, bw, bh, t, nd)
                n.grow(x, hy - bh - 0.2 * bh, x + bw, hy)
                x += bw + gap

    def _windows(self, node, x, y, w, h, t, nd):
        if not self._in_bucket(node):
            return
        rng = self.rng
        cols = max(1, int(w / rng.uniform(13, 24)))
        rows = max(1, int(h / rng.uniform(15, 30)))
        cw, ch = w / cols, h / rows
        # 整层亮/暗：真实办公楼是按层开灯的，不是逐格抽奖
        floor_lit = [rng.random() < (0.42 * t + 0.10) for _ in range(rows)]
        for r in range(rows):
            lit = floor_lit[r]
            if not lit and rng.random() > 0.28:
                continue
            for c in range(cols):
                if rng.random() > (0.68 if lit else 0.20):
                    continue
                ww = cw * rng.uniform(0.26, 0.62)      # 窗宽有户型差异
                wh = ch * rng.uniform(0.26, 0.46)
                wx = x + c * cw + (cw - ww) * rng.uniform(0.15, 0.85)
                wy = y + r * ch + (ch - wh) * rng.uniform(0.40, 0.88)
                col = lit_color("#ffd9a0", wx / self.w, wy / self.h, nd, self.light)
                self.body.append(f'<rect x="{wx:.1f}" y="{wy:.1f}" width="{ww:.1f}" '
                                 f'height="{wh:.1f}" fill="{col}" '
                                 f'opacity="{rng.uniform(0.28, 0.85):.2f}"/>')

    def ground(self, n):
        if not self._in_bucket(n):
            return
        p = n.params
        y = p.get("y", 0.62)
        color = p.get("color", "#1a1f26")
        self._rec(n, 0, y * self.h, self.w, self.h, "ground", "rect")
        if p.get("silhouette"):
            # 纯暗剪影地面：不加渐变、不吃光照。
            # 否则 lit_color 的 clamp（下限 0.16）会把柏油路抬成一堵亮灰墙，
            # 扩散看到「大块平面 + 中灰」就会自行补成混凝土墙。
            n.grow(0, y * self.h, self.w, self.h)
            self.body.append(f'<rect x="0" y="{y*self.h:.1f}" width="{self.w}" '
                             f'height="{(1-y)*self.h+2:.1f}" fill="{color}"/>')
            return
        if self.mode == "beauty":
            # 地面要压暗：lit_color 的 clamp 下限（0.16）会把柏油路抬成「一堵亮灰墙」。
            # 真实日落照片的地面亮度在 0.05–0.12。
            c_top = shade(lit_color(shade(color, 1.35), 0.5, y, n.depth, self.light), 0.58)
            c_bot = shade(lit_color(color, 0.5, 1.0, n.depth, self.light), 0.58)
            gid = self.gradient(c_top, c_bot, 0, y, 0, 1)
            n.grow(0, y * self.h, self.w, self.h)
            pid = None
            if self.mat:
                aid = self._material_of(n)
                pid = self._pattern_for(aid, self._mat_tile_px(n) / 256.0) if aid else None
            if pid:
                self.body.append(f'<rect x="0" y="{y*self.h:.1f}" width="{self.w}" '
                                 f'height="{(1-y)*self.h+2:.1f}" fill="url(#{pid})" opacity="0.9"/>')
            self.body.append(f'<rect x="0" y="{y*self.h:.1f}" width="{self.w}" '
                             f'height="{(1-y)*self.h+2:.1f}" fill="url(#{gid})" '
                             f'opacity="{MAT_LIGHT_OP if pid else 1.0}"/>')
            if p.get("texture", True):
                rng = random.Random(p.get("seed", 7))
                gtop = y * self.h
                # G3：旧版是 160 条横向长条（20–160px 宽、1–3px 高）→ 放大看是"扫描线"。
                # 真实沥青/铺装是「颗粒 + 接缝 + 局部水渍」，三种尺度分开画。
                # ① 颗粒：小方块（1–3px），密度高但对比低
                for _ in range(2200):
                    gx = rng.uniform(0, self.w)
                    gy = rng.uniform(gtop, self.h)
                    s2 = rng.uniform(1.0, 3.5)
                    self.body.append(f'<rect x="{gx:.1f}" y="{gy:.1f}" width="{s2:.1f}" '
                                     f'height="{s2:.1f}" fill="{shade(color, rng.uniform(1.2, 2.2))}" '
                                     f'opacity="{rng.uniform(0.02, 0.07):.3f}"/>')
                # ② 接缝：横向，间距随 y 递减（近密远疏）；只画 3–5 条，避免"扫描线"
                yy = gtop + (self.h - gtop) * rng.uniform(0.10, 0.22)
                while yy < self.h:
                    w2 = rng.uniform(1.0, 2.0)
                    self.body.append(f'<rect x="0" y="{yy:.1f}" width="{self.w}" '
                                     f'height="{w2:.1f}" fill="{shade(color, 0.75)}" '
                                     f'opacity="{rng.uniform(0.10, 0.18):.2f}"/>')
                    yy += (self.h - gtop) * rng.uniform(0.18, 0.34)
                # ③ 水渍：3–6 个大而极淡的椭圆（低对比，只改局部亮度）
                for _ in range(rng.randint(3, 6)):
                    cx2 = rng.uniform(0, self.w)
                    cy2 = rng.uniform(gtop + 0.15 * (self.h - gtop), self.h)
                    rx = rng.uniform(0.06, 0.18) * self.w
                    ry = rng.uniform(0.012, 0.035) * self.h
                    self.body.append(f'<ellipse cx="{cx2:.1f}" cy="{cy2:.1f}" rx="{rx:.1f}" '
                                     f'ry="{ry:.1f}" fill="{shade(color, 0.72)}" '
                                     f'opacity="{rng.uniform(0.08, 0.16):.2f}"/>')
        else:
            self.emit_rect(n, 0, y * self.h, self.w, (1 - y) * self.h + 2, color, hint="ground")

    def landmark(self, n):
        p = n.params
        if getattr(n, "material", None) is None:
            n.material = p.get("material", "glass")
        x = p.get("x", 0.62); horizon = p.get("horizon", 0.72)
        hh = p.get("h", 0.46); ww = p.get("w", 0.030)
        color = p.get("color", "#2a3b4d")
        cx = x * self.w
        top = (horizon - hh) * self.h
        basep = horizon * self.h
        bw = ww * self.w
        if p.get("taper", True):
            pts = [(cx - bw / 2, basep), (cx - bw * 0.28, top),
                   (cx + bw * 0.28, top), (cx + bw / 2, basep)]
            self.emit_poly(n, pts, color, hint="vertical", opacity=0.97)
        else:
            self.emit_rect(n, cx - bw / 2, top, bw, basep - top, color, opacity=0.97)
        ant = p.get("antenna", 0.06)
        if ant > 0:
            aw = max(2.0, bw * 0.06)
            self.emit_rect(n, cx - aw / 2, top - ant * self.h, aw, ant * self.h + 2, color, opacity=0.95)
        if self.mode == "beauty" and self._in_bucket(n):
            floors = p.get("floors", 26)
            rng = random.Random(p.get("seed", self.s.seed) ^ 0x5A5A)
            for i in range(1, floors):
                if rng.random() < 0.12:            # 设备层/避难层：无窗带
                    continue
                fy = top + (basep - top) * i / floors
                fw = bw * rng.uniform(0.44, 0.56)
                fx = cx - fw / 2 + rng.uniform(-0.01, 0.01) * bw
                self.body.append(f'<rect x="{fx:.1f}" y="{fy:.1f}" width="{fw:.1f}" '
                                 f'height="{max(0.8,(basep-top)/floors*rng.uniform(0.07,0.13)):.1f}" '
                                 f'fill="{shade(color,1.35)}" opacity="{rng.uniform(0.18,0.32):.2f}"/>')

    def trees(self, n):
        p = n.params
        y = p.get("y", 0.80); cnt = p.get("count", 9)
        size = p.get("size", (0.05, 0.11))
        color = p.get("color", "#0d1418")
        spread = p.get("spread", (0.0, 1.0))
        jit = p.get("jitter", 1.0)
        n.params["silhouette"] = p.get("silhouette", True)   # 剪影（不吃光照）
        rng = random.Random(p.get("seed", self.s.seed))
        for _ in range(cnt):
            cx = rng.uniform(spread[0], spread[1]) * self.w
            hh = rng.uniform(*size) * self.h
            r = hh * rng.uniform(0.50, 0.95)
            basep = y * self.h
            lean = rng.uniform(-0.10, 0.10) * r * jit        # 树干不垂直
            tw = max(2.0, r * rng.uniform(0.06, 0.14))
            self.emit_poly(n, [(cx - tw / 2, basep), (cx + tw / 2, basep),
                               (cx + tw / 2 + lean, basep - hh * 0.52),
                               (cx - tw / 2 + lean, basep - hh * 0.52)],
                           color, hint="vertical", opacity=0.95)
            # 树冠：叶簇（G1）。
            # 旧版是 4–8 个大椭圆叠加 → 放大后能看见圆的轮廓，像"黑色气球"。
            # 现在用 26–42 个小簇 + 面积均匀采样，轮廓变毛糙；每簇按"上部受光"微亮，
            # 仍然保持剪影（不吃场景光），只让剪影内部有层次。
            lobes = rng.randint(14, 22)
            cy0 = basep - hh * 0.62
            # 底衬：一个略小的实心椭圆，避免叶簇之间漏出背景（会被看成"破洞"）
            self.emit_circle(n, cx + lean, cy0, r * 0.72, shade(color, 0.9),
                             hint="foliage", opacity=0.97, blur=0.85)
            for _k in range(lobes):
                ang = rng.uniform(0, 2 * math.pi)
                rr = math.sqrt(rng.uniform(0.0, 1.0))     # 圆盘内面积均匀
                ox = math.cos(ang) * r * 0.64 * rr
                oy = math.sin(ang) * r * 0.52 * rr
                # 上亮下暗：逆光树冠顶部有天光透射，底部几乎全黑。
                top = max(0.0, -oy / max(r * 0.5, 1e-6))
                lit = 0.85 + 1.10 * top
                col = shade(color, lit * rng.uniform(0.92, 1.12))
                # 顶部叶缘透光（G1b）：逆光树叶会把太阳色透出来，没有这层就只是黑气球。
                # 只作用于最上缘 25%，强度随高度；白天场景 sun 色接近白，同样成立。
                # 透光改成「整体随高度渐变」，不再随机挑几个簇：
                # 挑簇会产生孤立的浅色圆点（放大看像果实/病斑），渐变才是叶缘透光的样子。
                col = mix(col, self.light.color, 0.04 + 0.13 * top)
                self.emit_circle(n, cx + lean + ox, cy0 + oy,
                                 r * rng.uniform(0.14, 0.30), col,
                                 hint="foliage", opacity=rng.uniform(0.88, 0.97))

    def railing(self, n):
        p = n.params
        y = p.get("y", 0.90); color = p.get("color", "#8d939a")
        gap = p.get("gap", 0.055); hh = p.get("height", 0.10)
        op = p.get("opacity", 0.55)
        jit = p.get("jitter", 1.0)
        rng = random.Random(p.get("seed", self.s.seed))
        basep = y * self.h; ph = hh * self.h
        x = 0.0
        while True:
            x += gap * self.w * (1 + rng.uniform(-0.18, 0.18) * jit)
            if x >= self.w:
                break
            if rng.random() < 0.07 * jit:            # 偶发缺柱
                continue
            pw = max(2.0, self.w * 0.0035 * (1 + rng.uniform(-0.30, 0.55) * jit))
            hh2 = ph * (1 + rng.uniform(-0.05, 0.02) * jit)
            self.emit_rect(n, x, basep - hh2, pw, hh2, color, hint="vertical",
                           opacity=min(1.0, op * (1 + rng.uniform(-0.18, 0.18) * jit)))
        # 横杆：分段、每段轻微错高 → 一眼看去「不笔直」
        seg = self.w / 26.0
        th = max(2.0, self.h * 0.004)
        for i in range(26):
            dy = rng.uniform(-0.9, 0.9) * jit
            self.emit_rect(n, i * seg, basep - ph + dy, seg * 1.03, th, color,
                           opacity=min(1.0, op * (1 + rng.uniform(-0.12, 0.12) * jit)))

    def rect(self, n):
        p = n.params
        self.emit_rect(n, p.get("x", 0.5) * self.w, p.get("y", 0.5) * self.h,
                       p.get("w", 0.2) * self.w, p.get("h", 0.2) * self.h,
                       p.get("fill", "#888888"), hint=p.get("hint", "vertical"),
                       opacity=p.get("opacity", 1.0))

    def circle(self, n):
        p = n.params
        self.emit_circle(n, p.get("cx", p.get("x", 0.5)) * self.w,
                         p.get("cy", p.get("y", 0.5)) * self.h,
                         p.get("r", 0.1) * self.h, p.get("fill", "#888888"),
                         hint=p.get("hint", "foliage"), opacity=p.get("opacity", 1.0))

    def poly(self, n):
        p = n.params
        pts = [(a * self.w, b * self.h) for a, b in p.get("pts", [])]
        self.emit_poly(n, pts, p.get("fill", "#888888"),
                       hint=p.get("hint", "vertical"), opacity=p.get("opacity", 1.0))

    # ---- 后处理（仅 beauty）----
    def grain(self, n):
        if self.mode != "beauty":
            return
        p = n.params
        fid = self._id("grain")
        self.defs.append(
            f'<filter id="{fid}" x="0" y="0" width="100%" height="100%">'
            f'<feTurbulence type="fractalNoise" baseFrequency="{0.9*p.get("scale",1.0):.3f}" '
            f'numOctaves="3" seed="{p.get("seed",7)}" result="n"/>'
            f'<feColorMatrix in="n" type="saturate" values="0"/></filter>')
        self.body.append(f'<rect width="{self.w}" height="{self.h}" filter="url(#{fid})" '
                         f'opacity="{p.get("amount",0.055)}" style="mix-blend-mode:overlay"/>')

    def vignette(self, n):
        if self.mode != "beauty":
            return
        p = n.params
        gid = self._id("vig")
        self.defs.append(
            f'<radialGradient id="{gid}" cx="0.5" cy="0.5" r="0.75">'
            f'<stop offset="0.55" stop-color="#000" stop-opacity="0"/>'
            f'<stop offset="1" stop-color="#000" stop-opacity="{p.get("strength",0.35)}"/>'
            f'</radialGradient>')
        self.body.append(f'<rect width="{self.w}" height="{self.h}" fill="url(#{gid})"/>')

    def blur_layer(self, n):
        if self.mode != "beauty" or not self.body:
            return
        gid = self._id("blur")
        px = n.params.get("blur_px", 6)
        self.defs.append(f'<filter id="{gid}"><feGaussianBlur stdDeviation="{px}"/></filter>')
        self.body = [f'<g filter="url(#{gid})">' + "\n".join(self.body) + "</g>"]

    def text(self, n):
        if not self._in_bucket(n):
            return
        p = n.params
        s = str(p.get("s", "")).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        fill = self.style(n, p.get("x", 0.5), p.get("y", 0.5), "vertical",
                          p.get("fill", "#ffffff"))
        self.body.append(f'<text x="{p.get("x",0.5)*self.w:.1f}" y="{p.get("y",0.5)*self.h:.1f}" '
                         f'font-size="{p.get("size",32)}" fill="{fill}" '
                         f'font-weight="{p.get("weight",400)}" '
                         f'font-family="Noto Sans CJK SC">{s}</text>')

    # ---- 输出 ----
    def compile(self):
        for n in self.s.nodes:
            fn = getattr(self, n.kind, None)
            if fn is None:
                raise SystemExit(f"未知图元: {n.kind}")
            if self.bucket is not None and n.kind in BACKGROUND_KINDS:
                continue          # 背景/后处理不进分层，单独处理
            fn(n)
            n.close_bbox()
        bg = "#ffffff" if self.mode == "lineart" else "none"
        self.s.mat_assign = {**getattr(self.s, "mat_assign", {}), **self.mat_assign}
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
                f'viewBox="0 0 {self.w} {self.h}">\n<defs>{"".join(self.defs)}</defs>\n'
                + "\n".join(self.body) + "\n</svg>"), self.seg_map


def compile_scene(scene, mode="beauty", bucket=None):
    c = Compiler(scene, mode, bucket=bucket)
    return c.compile()


# ================================================================ 渲染


def to_win(p):
    p = str(Path(p).resolve())
    if p.startswith("/mnt/") and len(p) > 6:
        return p[5].upper() + ":\\" + p[7:].replace("/", "\\")
    return p


def render_svg(svg, out_png, w, h, transparent=True):
    TMP.mkdir(parents=True, exist_ok=True)
    tag = abs(hash(svg)) % 10 ** 8
    html = TMP / f"s{tag}.html"
    html.write_text(
        f'<!doctype html><html><head><meta charset="utf-8">'
        f'<style>html,body{{margin:0;padding:0;overflow:hidden}}svg{{display:block}}</style>'
        f'</head><body>{svg}</body></html>', encoding="utf-8")
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    if out_png.exists():
        out_png.unlink()
    cmd = [CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
           "--force-device-scale-factor=1", f"--window-size={w},{h}",
           f"--screenshot={to_win(out_png)}",
           "--default-background-color=" + ("00000000" if transparent else "ffffffff"),
           "--virtual-time-budget=2500",
           f"file:///{to_win(html).replace(chr(92), '/')}"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if not out_png.exists():
        raise SystemExit(f"渲染失败: {r.stderr[-300:]}")
    return out_png


# ================================================================ Blender 导出

BLENDER_TMPL = '''# 由 procgen2 生成 —— Blender 4.4+（Blackwell/sm_120 需 4.4 起）
import bpy, math, json, os

W, H = {w}, {h}
CAM = {camera}
LIGHT = {light}
NODES = {nodes_json}

for ob in list(bpy.data.objects):
    bpy.data.objects.remove(ob, do_unlink=True)

scn = bpy.context.scene
scn.render.engine = 'CYCLES'
scn.cycles.device = 'GPU'
scn.cycles.samples = 64
scn.render.resolution_x, scn.render.resolution_y = W, H
scn.render.film_transparent = False

# 相机：按真实焦距/传感器换算视场角
cam_data = bpy.data.cameras.new('Cam')
cam_data.lens = CAM['focal']
cam_data.sensor_width = CAM['sensor']
cam_data.dof.use_dof = True
cam_data.dof.aperture_fstop = CAM['fstop']
cam = bpy.data.objects.new('Cam', cam_data)
scn.collection.objects.link(cam)
cam.location = (0, -12, 1.2)
cam.rotation_euler = (math.radians(88 + CAM['pitch']), 0, 0)
scn.camera = cam

# 太阳：方位角/高程角/色温
sun_data = bpy.data.lights.new('Sun', type='SUN')
sun_data.energy = 3.0 * LIGHT['intensity']
kel = LIGHT['kelvin']
# 黑体色温近似
def kelvin_rgb(k):
    t = k / 100.0
    r = 255 if t <= 66 else 329.698727446 * ((t - 60) ** -0.1332047592)
    g = (99.4708025861 * math.log(t) - 161.1195681661) if t <= 66 else 288.1221695283 * ((t - 60) ** -0.0755148492)
    b = 255 if t >= 66 else (0 if t <= 19 else 138.5177312231 * math.log(t - 10) - 305.0447927307)
    f = lambda v: max(0.0, min(1.0, v / 255.0))
    return (f(r) ** 2.2, f(g) ** 2.2, f(b) ** 2.2, 1.0)
sun_data.color = kelvin_rgb(kel)[:3]
sun = bpy.data.objects.new('Sun', sun_data)
scn.collection.objects.link(sun)
az, el = math.radians(LIGHT['azimuth']), math.radians(LIGHT['elevation'])
sun.rotation_euler = (math.radians(90) - el, 0, az)

world = bpy.data.worlds.new('W') if not scn.world else scn.world
scn.world = world
world.use_nodes = True
bg = world.node_tree.nodes.get('Background')
if bg:
    bg.inputs[0].default_value = kelvin_rgb(6500)
    bg.inputs[1].default_value = LIGHT['ambient']

# 每个节点一个平面卡（2.5D），z 由 depth 决定
for i, n in enumerate(NODES):
    if not n.get('bbox'):
        continue
    x0, y0, x1, y1 = n['bbox']
    cx = (x0 + x1) / 2.0 / W
    cy = (y0 + y1) / 2.0 / H
    bw = max(0.02, (x1 - x0) / W)
    bh = max(0.02, (y1 - y0) / H)
    mesh = bpy.data.meshes.new(f"m{{i}}")
    mesh.from_pydata([(-1,-1,0),(1,-1,0),(1,1,0),(-1,1,0)], [], [(0,1,2,3)])
    mesh.update()
    ob = bpy.data.objects.new(n['id'], mesh)
    scn.collection.objects.link(ob)
    ob.location = ((cx - 0.5) * 2.0 * (12.0 / CAM['focal']) * 8.0,
                   (0.5 - cy) * 2.0 * (12.0 / CAM['focal']) * 8.0 * (H / W) * (W / H),
                   -n['depth'] * 10.0)
    ob.scale = (bw * 8.0, bh * 8.0, 1.0)
    tex = n.get('tex')
    mat = bpy.data.materials.new(f"mat{{i}}")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    if bsdf:
        bsdf.inputs['Base Color'].default_value = n.get('color', (0.5, 0.5, 0.5, 1.0))
        bsdf.inputs['Roughness'].default_value = n.get('rough', 0.7)
    if tex and os.path.exists(tex):
        img = bpy.data.images.load(tex)
        tn = mat.node_tree.nodes.new('ShaderNodeTexImage')
        tn.image = img
        mat.node_tree.links.new(tn.outputs['Color'], bsdf.inputs['Base Color'])
        mat.blend_method = 'BLEND' if hasattr(mat, 'blend_method') else mat.blend_method
    ob.data.materials.append(mat)

# G-buffer 输出：View Layer passes
vl = scn.view_layers[0]
vl.use_pass_z = True
vl.use_pass_normal = True
vl.use_pass_mist = True
vl.use_pass_environment = True
scn.render.image_settings.color_mode = 'RGBA'
print('scene ready:', len(NODES), 'cards')
'''


def to_blender_py(scene, mode_compile=None):
    """导出 Blender 场景脚本。贴图路径可由 node.params['tex'] 指定。

    先编译一遍以填充 bbox（否则没有任何几何）。
    """
    compile_scene(scene, "beauty")
    SKIP = {"sky", "vignette", "grain", "blur", "haze", "sun", "text"}

    def node_color(n):
        for k in ("fill", "color", "base_color", "top"):
            v = n.params.get(k)
            if isinstance(v, str) and v.startswith("#"):
                return [c / 255.0 for c in hex2rgb(v)] + [1.0]
        return [0.5, 0.5, 0.5, 1.0]

    nodes = []
    for n in scene.nodes:
        if n.kind in SKIP:
            continue
        nodes.append(dict(
            id=n.id, bbox=list(n.bbox) if n.bbox else None,
            depth=n.depth,
            color=node_color(n),
            rough=MATERIALS.get(n.material or "concrete", {}).get("rough", 0.75),
            tex=n.params.get("tex"),
        ))
    return BLENDER_TMPL.format(
        w=scene.w, h=scene.h,
        camera=json.dumps(scene.camera, ensure_ascii=False),
        light=json.dumps({k: v for k, v in scene.light.__dict__.items()
                          if isinstance(v, (int, float, str))}, ensure_ascii=False),
        nodes_json=json.dumps(nodes, ensure_ascii=False, indent=1),
    )


# ================================================================ 主流程


DEMO = {
    "w": 1920, "h": 1080, "seed": 20261025,
    "light": {"azimuth": 258, "elevation": 9, "kelvin": 2700,
              "intensity": 1.15, "ambient": 0.30, "haze": 0.38},
    "camera": {"focal": 50, "fstop": 2.8},
    "nodes": [
        {"kind": "sky", "depth": 0.05, "params": {"top": "#2b5a8c", "bottom": "#f6d9a8", "horizon": 0.60}},
        {"kind": "sun", "depth": 0.08, "params": {"x": 0.74, "y": 0.40, "r": 0.055, "glow": 3.2}},
        {"kind": "haze", "depth": 0.20, "params": {"y": 0.56, "height": 0.22, "opacity": 0.42}},
        {"kind": "skyline", "depth": 0.55, "params": {"horizon": 0.60, "depth": 4,
                                                      "base_color": "#33465c", "window_p": 0.30,
                                                      "height_range": [0.10, 0.42], "crowd": 0.85,
                                                      "seed": 11}},
        {"kind": "ground", "depth": 0.92, "params": {"y": 0.615, "color": "#141a21", "seed": 5}},
        {"kind": "landmark", "depth": 0.68, "params": {"x": 0.30, "horizon": 0.615, "h": 0.40,
                                                       "w": 0.026, "color": "#26333f", "floors": 30}},
        {"kind": "trees", "depth": 0.97, "params": {"y": 0.98, "count": 7, "size": [0.07, 0.13],
                                                     "color": "#080c10", "spread": [0.02, 0.42], "seed": 3}},
        {"kind": "vignette", "depth": 1.0, "params": {"strength": 0.30}},
        {"kind": "grain", "depth": 1.0, "params": {"amount": 0.05, "scale": 1.0, "seed": 7}},
    ],
}


def scene_from_dict(d):
    if "ops" in d and "nodes" not in d:
        return from_v1_ops(d["ops"], d.get("w", 1920), d.get("h", 1080),
                           d.get("seed", 42), d.get("light"))
    sc = Scene(d.get("w", 1920), d.get("h", 1080), d.get("seed", 42),
               light=d.get("light"), camera=d.get("camera"))
    for nd in d.get("nodes", []):
        sc.add(nd["kind"], nd.get("params"), nid=nd.get("id"),
               depth=nd.get("depth", 0.5), material=nd.get("material"))
    apply_layout(sc, d.get("layout"))
    return sc


def build_all(scene, outdir, modes, prefix, transparent=True):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    seg_map = None
    written = []
    for m in modes:
        svg, sm = compile_scene(scene, m)
        if sm:
            seg_map = sm
        png = outdir / f"{prefix}_{m}.png"
        render_svg(svg, png, scene.w, scene.h,
                   transparent=(transparent and m in ("beauty", "seg", "lineart")))
        written.append(str(png))
        if m == "beauty":
            (outdir / f"{prefix}.svg").write_text(svg, encoding="utf-8")
    meta = dict(
        w=scene.w, h=scene.h, seed=scene.seed,
        camera=scene.camera,
        light={k: v for k, v in scene.light.__dict__.items()
               if isinstance(v, (int, float, str))},
        nodes=[dict(id=n.id, kind=n.kind, depth=n.depth,
                    bbox=list(n.bbox) if n.bbox else None)
               for n in scene.nodes],
        materials=getattr(scene, "mat_assign", {}),
        seg=seg_map or {},
    )
    (outdir / f"{prefix}_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return written


def build_layers(scene, outdir, prefix, n_layers=4):
    """按 depth 切层，输出可合成/可位移的 2.5D 分层。

    层 0 = 背景（sky/sun/haze/后处理），不透明；
    层 1..N = 按 depth 等分，带 alpha，越靠前视差系数越大。
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    meta = dict(w=scene.w, h=scene.h, layers=[])

    bg = Scene(scene.w, scene.h, scene.seed)
    bg.light, bg.camera = scene.light, scene.camera
    bg.nodes = [n for n in scene.nodes if n.kind in BACKGROUND_KINDS]
    if bg.nodes:
        svg, _ = compile_scene(bg, "beauty")
        p = outdir / f"{prefix}_layer0_bg.png"
        render_svg(svg, p, scene.w, scene.h, transparent=False)
        meta["layers"].append(dict(index=0, kind="background", depth=[0.0, 0.0],
                                   parallax=0.0, file=p.name))

    edges = [i / n_layers for i in range(n_layers + 1)]
    for i in range(n_layers):
        lo, hi = edges[i], edges[i + 1]
        svg, _ = compile_scene(scene, "beauty", bucket=(lo, hi))
        if "<rect" not in svg and "<polygon" not in svg and "<circle" not in svg:
            continue                       # 空层跳过
        p = outdir / f"{prefix}_layer{i+1}_d{lo:.2f}-{hi:.2f}.png"
        render_svg(svg, p, scene.w, scene.h, transparent=True)
        meta["layers"].append(dict(
            index=i + 1, kind="content", depth=[round(lo, 3), round(hi, 3)],
            parallax=round((lo + hi) / 2, 3), file=p.name))

    (outdir / f"{prefix}_layers.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


DEFAULT_GAP = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "baseline", "gap_report_realistic.json")


def _enable_matpick(sc, use_matpick, gap_path):
    """v2 开关：载入指标缺口，启用打分选型。失败时保持哈希路径。"""
    if not use_matpick:
        return
    path = gap_path or DEFAULT_GAP
    if not os.path.exists(path):
        print(f"[matpick] 缺 {path}，保持哈希选型")
        return
    if not sc.load_gap(path):
        print("[matpick] 无 nodes.jsonl（材质节点），保持哈希选型")
        return



def main(argv):
    args = argv[1:]
    if not args:
        print(__doc__)
        return 2
    modes = ["beauty"]
    outdir = None
    blender_out = None
    src = None
    batch = None
    n_layers = 0
    mat_root = None
    use_matpick = True     # v2 默认启用（缺 nodes.jsonl / gap 时自动回退哈希）
    gap_path = None
    tile_grad = 0.0
    mat_topk = None
    mat_res = "auto"
    mat_tile_scale = MAT_TILE_SCALE
    albedo_w = 0.0
    albedo_ref = 0.18

    i = 0
    while i < len(args):
        a = args[i]
        if a == "--demo":
            src = "__demo__"
        elif a == "--gbuffer":
            i += 1
            v = args[i]
            modes = list(MODES) if v == "all" else [m.strip() for m in v.split(",") if m.strip()]
        elif a == "--blender":
            i += 1
            blender_out = args[i]
        elif a == "--layers":
            i += 1
            n_layers = int(args[i])
        elif a == "--batch":
            i += 1
            batch = args[i]
        elif a == "--materials":
            i += 1
            mat_root = args[i]
        elif a == "--matpick":
            use_matpick = True
        elif a == "--no-matpick":
            use_matpick = False
        elif a == "--gap":
            i += 1
            gap_path = args[i]
        elif a == "--tile-grad":
            i += 1
            tile_grad = float(args[i])
        elif a == "--mat-topk":
            i += 1
            mat_topk = int(args[i])
        elif a == "--mat-res":
            i += 1
            mat_res = args[i]
            if mat_res not in ("auto", "256", "512", "1024", "orig"):
                raise SystemExit(f"--mat-res 只接受 auto/256/512/1024/orig，收到 {mat_res!r}")
        elif a == "--tile-scale":
            i += 1
            mat_tile_scale = float(args[i])
        elif a == "--albedo-w":
            i += 1
            albedo_w = float(args[i])
        elif a == "--albedo-ref":
            i += 1
            albedo_ref = float(args[i])
        elif a.startswith("--"):
            print(f"未知参数 {a}")
            return 2
        else:
            if src is None and a.endswith(".json"):
                src = a
            elif outdir is None:
                outdir = a
        i += 1

    if batch:
        files = sorted(Path(batch).glob("*.json"))
        outdir = Path(outdir or TMP)
        ok = 0
        for f in files:
            sc = scene_from_dict(json.loads(f.read_text(encoding="utf-8")))
            if mat_root:
                sc.load_materials(mat_root)
            sc.tile_grad = tile_grad
            sc.mat_res = mat_res
            sc.mat_tile_scale = mat_tile_scale
            sc.albedo_w = albedo_w
            sc.albedo_ref = albedo_ref
            if mat_topk is not None:
                sc.mat_topk = mat_topk
            _enable_matpick(sc, use_matpick, gap_path)
            build_all(sc, outdir, modes, f.stem)
            ok += 1
            print(f"  [{ok}/{len(files)}] {f.name}")
        print(f"批量完成 {ok} 个场景 -> {outdir}")
        return 0

    scene = scene_from_dict(DEMO) if src == "__demo__" else scene_from_dict(
        json.loads(Path(src).read_text(encoding="utf-8")))
    if mat_root:
        scene.load_materials(mat_root)
    scene.tile_grad = tile_grad
    scene.mat_res = mat_res
    scene.mat_tile_scale = mat_tile_scale
    scene.albedo_w = albedo_w
    scene.albedo_ref = albedo_ref
    if mat_topk is not None:
        scene.mat_topk = mat_topk
    _enable_matpick(scene, use_matpick, gap_path)
    outdir = Path(outdir or TMP)

    if blender_out:
        Path(blender_out).write_text(to_blender_py(scene), encoding="utf-8")
        print(f"Blender 脚本 -> {blender_out}")
        return 0

    if n_layers > 0:
        meta = build_layers(scene, outdir, "scene", n_layers)
        print(f"2.5D 分层完成：{len(meta['layers'])} 层 -> {outdir}")
        for L in meta["layers"]:
            print(f"   L{L['index']} depth={L['depth']} parallax={L['parallax']} {L['file']}")
        return 0

    files = build_all(scene, outdir, modes, "scene", transparent=True)
    print(f"OK {len(files)} 通道 -> {outdir}")
    for f in files:
        print("  ", f)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
