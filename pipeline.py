#!/usr/bin/env python3
"""pipeline.py — 出图唯一入口（P0 接线）

约束（见 13-管线接线计划）：
  1) 成品只能由本脚本产出；make_wallpapers.py 已降级为薄封装。
  2) 每阶段声明 inputs/outputs/params，落 manifest.json；缺产物即失败。
  3) 能力默认开启，用 --no-<stage> / 参数置 0 关闭；关闭项写入 capabilities_off。

用法：
    pipeline.py run baseline/scenes/night_blue.json
    pipeline.py run SCENE --no-gen            # 无 ComfyUI 时只跑本地渲染链
    pipeline.py run SCENE --from shade        # 断点重跑
    pipeline.py check                         # 接线自检（W1-W5）
    pipeline.py list                          # 能力 × 状态 × 调用点
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROCGEN = Path(__file__).resolve().parent
VENV = Path(os.environ.get("PROCGEN_PYTHON", sys.executable))
MAT = Path(os.environ.get("PROCGEN_MATERIALS", str(Path.home() / ".cache" / "procgen" / "materials")))
CAPS = PROCGEN / "capabilities.json"
WIN_TMP = Path(os.environ.get("PROCGEN_TMP", str(Path.home() / ".cache" / "procgen" / "pipe")))
STAGES = ["build", "zbuf", "raster", "shade", "gen", "degrade"]
COMFY = "http://127.0.0.1:8188/system_stats"


def sha(p):
    p = Path(p)
    if not p.exists() or p.is_dir():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def run(cmd, tag):
    t0 = datetime.now()
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    dt = (datetime.now() - t0).total_seconds()
    if r.returncode != 0:
        tail = (r.stderr or r.stdout or "")[-600:]
        raise SystemExit(f"[{tag}] 失败 ({r.returncode})\n{tail}")
    return dt, (r.stdout or "").strip()


def scene_sun(scene_path):
    """从场景 JSON 读太阳位置（归一化 x,y）；夜景无 sun 返回 None。"""
    d = json.loads(Path(scene_path).read_text(encoding="utf-8"))
    for op in d.get("ops", []):
        if op.get("type") == "sun" and op.get("x") is not None:
            return float(op["x"]), float(op.get("y", 0.15))
    for n in d.get("nodes", []):
        if n.get("kind") == "sun":
            pr = n.get("params") or {}
            if pr.get("x") is not None:
                return float(pr["x"]), float(pr.get("y", 0.15))
    return None


def comfy_up():
    import urllib.request
    try:
        urllib.request.urlopen(COMFY, timeout=2)
        return True
    except Exception:
        return False


def load_caps():
    return json.loads(CAPS.read_text(encoding="utf-8")) if CAPS.exists() else {"capabilities": []}


def cap_ids(status="active"):
    return [c["id"] for c in load_caps().get("capabilities", []) if c.get("status") == status]


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run")
    r.add_argument("scene")
    r.add_argument("--out")
    r.add_argument("--preset", default="photo")
    r.add_argument("--mat-res", default="auto")
    r.add_argument("--tile-grad", type=float, default=1.0)
    r.add_argument("--albedo-w", type=float, default=0.15)
    r.add_argument("--ao", type=float, default=0.55)
    r.add_argument("--contact", type=float, default=0.30)
    r.add_argument("--haze-pp", type=float, default=0.12)
    r.add_argument("--godray", type=float, default=0.065)
    r.add_argument("--godray-power", type=float, default=3.0)
    r.add_argument("--godray-radius", type=float, default=0.55)
    r.add_argument("--denoise", type=float, default=0.35)
    r.add_argument("--cn-depth", type=float, default=0.90)
    r.add_argument("--cn-lineart", type=float, default=0.45)
    r.add_argument("--steps", type=int, default=24)
    r.add_argument("--size", default="768x480")
    r.add_argument("--out-size", default=None, help="降级前重采样到目标尺寸，如 2560x1600")
    r.add_argument("--prompt", default="")
    r.add_argument("--from", dest="from_stage", choices=STAGES)
    r.add_argument("--to", dest="to_stage", choices=STAGES)
    r.add_argument("--no-zbuf", action="store_true")
    r.add_argument("--no-raster", action="store_true")
    r.add_argument("--raster-mode", choices=("overlay", "replace"), default="overlay")
    r.add_argument("--detail-strength", type=float, default=0.25)
    r.add_argument("--tex-blur", type=float, default=1.2)
    r.add_argument("--no-shade", action="store_true")
    r.add_argument("--no-gen", action="store_true")
    r.add_argument("--no-degrade", action="store_true")
    r.add_argument("--force", action="store_true", help="即使 8188 不可达也尝试 gen")

    c = sub.add_parser("check")
    c.add_argument("--manifest")
    sub.add_parser("list")
    args = ap.parse_args(argv)

    if args.cmd == "list":
        for cap in load_caps().get("capabilities", []):
            print(f"{cap['status']:<9} {cap['id']:<24} default={str(cap.get('default')):<5} "
                  f"producer={cap.get('producer','-'):<28} consumer={cap.get('consumer','-')}")
        return 0
    if args.cmd == "check":
        cmd = [VENV, PROCGEN / "check_wiring.py"]
        if args.manifest:
            cmd += ["--manifest", args.manifest]
        return subprocess.call([str(x) for x in cmd])

    # ---------------- run ----------------
    scene = Path(args.scene).resolve()
    if not scene.exists():
        raise SystemExit(f"场景不存在: {scene}")
    out = Path(args.out) if args.out else WIN_TMP / scene.stem
    if not str(out).startswith("/mnt/"):
        raise SystemExit("输出目录必须在 /mnt/<盘>/ 下（Chrome 看不到 WSL 的 /tmp）")
    out.mkdir(parents=True, exist_ok=True)

    disabled = set()
    if args.no_zbuf:
        disabled.add("zbuf")
    if args.no_raster:
        disabled.add("raster")
    if args.no_shade:
        disabled.add("shade")
    if args.no_gen:
        disabled.add("gen")
    if args.no_degrade:
        disabled.add("degrade")

    STAGE_CAPS = {
        "build": ["mat.tile_grad", "mat.albedo_fit"],
        "zbuf": ["zbuf.depth", "zbuf.normal", "zbuf.ao", "zbuf.contact", "zbuf.seg"],
        "raster": ["mat.albedo_overlay", "mat.roughness_ggx", "mat.normal_light"],
        "shade": ["shade.ao", "shade.contact", "shade.haze_pp", "shade.godray"],
        "gen": ["gen.controlnet_depth", "gen.controlnet_lineart", "gen.upscale"],
        "degrade": ["imaging.degrade"],
    }

    def enabled(stage):
        if stage in disabled:
            return False
        if args.from_stage and STAGES.index(stage) < STAGES.index(args.from_stage):
            return False
        if args.to_stage and STAGES.index(stage) > STAGES.index(args.to_stage):
            return False
        return True

    build, zbufd, raster, shaded, gen, deg = (out / s for s in
                                             ("build", "zbuf", "raster", "shade", "gen", "degrade"))
    used, off, stages = [], [], []
    t_start = datetime.now()

    # ---- 1 build ----
    if enabled("build"):
        build.mkdir(exist_ok=True)
        dt, _ = run([VENV, PROCGEN / "procgen2.py", scene, build,
                     "--gbuffer", "beauty,depth,normal,lineart,seg",
                     "--materials", MAT, "--mat-res", args.mat_res,
                     "--tile-grad", args.tile_grad, "--albedo-w", args.albedo_w],
                    "build")
        outs = {p.stem.replace("scene_", ""): p for p in build.glob("scene_*.png")}
        stages.append({"name": "build", "seconds": round(dt, 2),
                       "params": {"mat_res": args.mat_res, "tile_grad": args.tile_grad,
                                  "albedo_w": args.albedo_w, "gbuffer": "beauty,depth,normal,lineart,seg"},
                       "outputs": {k: {"path": str(v), "sha256": sha(v)} for k, v in outs.items()}})
        used.append("mat.tile_grad")
        if args.albedo_w > 0:
            used.append("mat.albedo_fit")

    # ---- 2 zbuf ----
    if enabled("zbuf"):
        zbufd.mkdir(exist_ok=True)
        dt, _ = run([VENV, PROCGEN / "zbuf.py", scene, "--out", zbufd, "--time"], "zbuf")
        outs = {p.stem: p for p in zbufd.glob("*.png")}
        stages.append({"name": "zbuf", "seconds": round(dt, 2), "params": {},
                       "outputs": {k: {"path": str(v), "sha256": sha(v)} for k, v in outs.items()}})
        used += ["zbuf.depth", "zbuf.normal", "zbuf.ao", "zbuf.contact", "zbuf.seg"]

    # ---- 2.5 raster（P1：逐像素材质细节/法线光照）----
    if enabled("raster"):
        raster.mkdir(exist_ok=True)
        meta_p = build / "scene_meta.json"
        if not (zbufd / "ids.npy").exists() or not meta_p.exists():
            raise SystemExit("[raster] 缺 zbuf 产物或 scene_meta.json")
        dt, _ = run([VENV, PROCGEN / "shade_raster.py", "--zdir", zbufd,
                     "--meta", meta_p, "--base", build / "scene_beauty.png",
                     "--materials", MAT, "--out", raster / "beauty.png",
                     "--mode", args.raster_mode,
                     "--detail-strength", args.detail_strength,
                     "--tex-blur", args.tex_blur], "raster")
        stages.append({"name": "raster", "seconds": round(dt, 2),
                       "params": {"mode": args.raster_mode,
                                  "detail_strength": args.detail_strength,
                                  "tex_blur": args.tex_blur},
                       "outputs": {"beauty": {"path": str(raster / "beauty.png"),
                                              "sha256": sha(raster / "beauty.png")}}})
        used += ["mat.albedo_overlay", "mat.roughness_ggx"]
        if args.raster_mode == "replace":
            used.append("mat.normal_light")

    # ---- 3 shade ----
    if enabled("shade"):
        shaded.mkdir(exist_ok=True)
        src = (gen / "r.png") if (gen / "r.png").exists() else (
            (raster / "beauty.png") if (raster / "beauty.png").exists()
            else (build / "scene_beauty.png"))
        if not src.exists():
            raise SystemExit("[shade] 缺 beauty（build 被跳过且无 gen 产物）")
        sun = scene_sun(scene)
        god = args.godray if sun else 0.0
        sun_arg = f"{sun[0]},{sun[1]}" if sun else "0.20,0.15"
        dt, _ = run([VENV, PROCGEN / "apply_ao.py", src, "--zdir", zbufd,
                     "--out", shaded / "scene_beauty.png",
                     "--ao", args.ao, "--contact", args.contact,
                     "--wet", 0, "--haze-pp", args.haze_pp, "--godray", god,
                     "--godray-power", args.godray_power,
                     "--godray-radius", args.godray_radius,
                     "--sun", sun_arg, "--report"], "shade")
        stages.append({"name": "shade", "seconds": round(dt, 2),
                       "params": {"ao": args.ao, "contact": args.contact, "wet": 0,
                                  "haze_pp": args.haze_pp, "godray": god,
                                  "godray_power": args.godray_power,
                                  "godray_radius": args.godray_radius,
                                  "sun": list(sun) if sun else None},
                       "outputs": {"beauty": {"path": str(shaded / "scene_beauty.png"),
                                              "sha256": sha(shaded / "scene_beauty.png")}}})
        used += ["shade.ao", "shade.contact"]
        if args.haze_pp > 0:
            used.append("shade.haze_pp")
        if god > 0:
            used.append("shade.godray")
        else:
            off.append("shade.godray")

    # ---- 4 gen ----
    if enabled("gen"):
        if not comfy_up() and not args.force:
            stages.append({"name": "gen", "skipped": True, "reason": "ComfyUI 8188 不可达"})
            off.extend(STAGE_CAPS["gen"])
        else:
            gen.mkdir(exist_ok=True)
            base = shaded / "scene_beauty.png"
            if not base.exists():
                base = build / "scene_beauty.png"
            depth = build / "scene_depth.png"          # 已否决 zbuf depth 替换（r=0.984）
            line = build / "scene_lineart.png"
            dt, _ = run([VENV, PROCGEN / "comfy_render.py", "--beauty", base,
                         "--depth", depth, "--lineart", line, "--out", gen / "r.png",
                         "--denoise", args.denoise, "--cn-depth", args.cn_depth,
                         "--cn-lineart", args.cn_lineart, "--size", args.size,
                         "--steps", args.steps,
                         "--prompt", args.prompt or "photorealistic city scene",
                         "--upscale-model", "RealESRGAN_x4.pth"], "gen")
            stages.append({"name": "gen", "seconds": round(dt, 2),
                           "params": {"denoise": args.denoise, "cn_depth": args.cn_depth,
                                      "cn_lineart": args.cn_lineart, "size": args.size,
                                      "steps": args.steps,
                                      "upscale": "RealESRGAN_x4.pth",
                                      "depth_source": "procgen2"},
                           "outputs": {"r": {"path": str(gen / "r.png"), "sha256": sha(gen / "r.png")}}})
            used += ["gen.controlnet_depth", "gen.controlnet_lineart", "gen.upscale"]

    # ---- 5 degrade ----
    if enabled("degrade"):
        deg.mkdir(exist_ok=True)
        src = gen / "r.png"
        if not src.exists():
            src = shaded / "scene_beauty.png"
        if not src.exists():
            src = build / "scene_beauty.png"
        if args.out_size:
            from PIL import Image
            tw, th = (int(x) for x in args.out_size.lower().split("x"))
            im = Image.open(src).convert("RGB").resize((tw, th), Image.Resampling.LANCZOS)
            pre = deg / "pre.png"
            im.save(pre)
            src = pre
        dt, _ = run([VENV, PROCGEN / "degrade.py", src, deg / "final.jpg",
                     "--preset", args.preset, "--report"], "degrade")
        stages.append({"name": "degrade", "seconds": round(dt, 2),
                       "params": {"preset": args.preset, "out_size": args.out_size},
                       "outputs": {"final": {"path": str(deg / "final.jpg"),
                                             "sha256": sha(deg / "final.jpg")}}})
        used.append("imaging.degrade")

    active = set(cap_ids("active"))
    used = sorted(set(used) & active)
    off = set(off)
    for s in disabled:
        off.update(STAGE_CAPS.get(s, []))
    off = sorted(off & active)
    camera = {}
    meta_p = build / "scene_meta.json"
    if meta_p.exists():
        camera = json.loads(meta_p.read_text(encoding="utf-8")).get("camera", {})
    manifest = {
        "run_id": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "scene": str(scene), "outdir": str(out),
        "git": subprocess.run(["git", "-C", str(PROCGEN), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip(),
        "seconds": round((datetime.now() - t_start).total_seconds(), 2),
        "stages": stages,
        "camera": camera,
        "capabilities_used": used,
        "capabilities_off": off,
        "disabled_stages": sorted(disabled),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"→ {out}/manifest.json")
    print(f"   stages={[s['name'] for s in stages]}")
    print(f"   used={used}")
    if off:
        print(f"   off={off}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
