#!/usr/bin/env python3
"""comfy_render.py — 把 G-buffer 送进 ComfyUI 做结构保持重渲染（SD1.5 + ControlNet）

为什么要动态探测节点字段：ControlNetApplyAdvanced 在 2024-2026 之间改过签名
（有的版本要 `vae`，有的不要；`strength` 有的拆成 start/end）。写死就会 400。
本脚本先 GET /object_info 拿真实 input 定义，再按存在性补齐。

用法：
    python comfy_render.py --beauty A.png --depth B.png --out C.png \
        --prompt "..." --denoise 0.32 --cn-depth 0.9 --size 768x432

    # 只用 lineart
    python comfy_render.py --beauty A.png --lineart C.png --out D.png --modes lineart
"""
import argparse
import json
import os
import shutil
import sys
import time
import urllib.parse
import urllib.request
import uuid

HOST = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")
COMFY = os.environ.get("COMFYUI_DIR", os.path.expanduser("~/ComfyUI"))

POS_DEFAULT = ("photorealistic photograph of a city skyline at sunset, real concrete and glass "
               "facades with visible surface texture, warm backlight, atmospheric haze, "
               "35mm lens, natural depth of field, fine detail")
NEG_DEFAULT = ("illustration, cartoon, flat vector, painting, cgi, 3d render, plastic, "
               "smooth gradient, oversaturated, blurry, low contrast, text, watermark, logo")


def http_json(path, payload=None, timeout=30):
    url = HOST + path
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def node_info(cls):
    try:
        d = http_json(f"/object_info/{cls}")
        return (d.get(cls) or {}).get("input", {}).get("required", {})
    except Exception:
        return {}


def stage(src, tag):
    """把图片放到 ComfyUI 的 input 目录（LoadImage 只能读那里）。"""
    inp = os.path.join(COMFY, "input")
    os.makedirs(inp, exist_ok=True)
    name = f"pg_{tag}_{uuid.uuid4().hex[:8]}" + os.path.splitext(src)[1].lower()
    shutil.copy(src, os.path.join(inp, name))
    return name


def build(beauty, ctrls, p, cfg):
    """ctrls: [(图片名, controlnet 文件名, strength)]"""
    wf = {
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": cfg["ckpt"]}},
        "2": {"class_type": "CLIPTextEncode",
              "inputs": {"text": p["prompt"], "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode",
              "inputs": {"text": p["negative"], "clip": ["1", 1]}},
        "4": {"class_type": "LoadImage", "inputs": {"image": beauty}},
        "5": {"class_type": "ImageScale",
              "inputs": {"image": ["4", 0], "upscale_method": "lanczos",
                         "width": p["w"], "height": p["h"], "crop": "disabled"}},
    }
    pos, neg = ["2", 0], ["3", 0]
    nid = 10
    for img, cn_name, strength in ctrls:
        a, b, c = str(nid), str(nid + 1), str(nid + 2)
        wf[a] = {"class_type": "LoadImage", "inputs": {"image": img}}
        wf[b] = {"class_type": "ImageScale",
                 "inputs": {"image": [a, 0], "upscale_method": "lanczos",
                            "width": p["w"], "height": p["h"], "crop": "disabled"}}
        wf[c] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": cn_name}}
        req = node_info("ControlNetApplyAdvanced")
        inputs = {"positive": pos, "negative": neg, "control_net": [c, 0],
                  "image": [b, 0], "strength": strength}
        if "start_percent" in req:
            inputs["start_percent"] = 0.0
        if "end_percent" in req:
            inputs["end_percent"] = cfg["cn_end"]
        if "vae" in req:                      # 新版需要 vae
            inputs["vae"] = ["1", 2]
        adv = str(nid + 3)
        wf[adv] = {"class_type": "ControlNetApplyAdvanced", "inputs": inputs}
        pos, neg = [adv, 0], [adv, 1]
        nid += 4

    wf[str(nid)] = {"class_type": "VAEEncode",
                    "inputs": {"pixels": ["5", 0], "vae": ["1", 2]}}
    wf[str(nid + 1)] = {"class_type": "KSampler",
                        "inputs": {"model": ["1", 0], "positive": pos, "negative": neg,
                                   "latent_image": [str(nid), 0], "seed": p["seed"],
                                   "steps": p["steps"], "cfg": p["cfg"],
                                   "sampler_name": p["sampler"], "scheduler": p["scheduler"],
                                   "denoise": p["denoise"]}}
    wf[str(nid + 2)] = {"class_type": "VAEDecode",
                        "inputs": {"samples": [str(nid + 1), 0], "vae": ["1", 2]}}
    last = str(nid + 2)
    if cfg.get("upscale_model"):
        u1, u2 = str(nid + 3), str(nid + 4)
        wf[u1] = {"class_type": "UpscaleModelLoader",
                  "inputs": {"model_name": cfg["upscale_model"]}}
        wf[u2] = {"class_type": "ImageUpscaleWithModel",
                  "inputs": {"upscale_model": [u1, 0], "image": [last, 0]}}
        last = u2
        nid += 2
    wf[str(nid + 3)] = {"class_type": "SaveImage",
                        "inputs": {"images": [last, 0],
                                   "filename_prefix": "procgen"}}
    return wf, str(nid + 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beauty", required=True)
    ap.add_argument("--depth")
    ap.add_argument("--lineart")
    ap.add_argument("--normal")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default=POS_DEFAULT)
    ap.add_argument("--negative", default=NEG_DEFAULT)
    ap.add_argument("--denoise", type=float, default=0.32)
    ap.add_argument("--cn-depth", type=float, default=0.90)
    ap.add_argument("--cn-lineart", type=float, default=0.75)
    ap.add_argument("--cn-normal", type=float, default=0.55)
    ap.add_argument("--cn-end", type=float, default=0.80)
    ap.add_argument("--size", default="768x432")
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--cfg", type=float, default=6.5)
    ap.add_argument("--seed", type=int, default=20261026)
    ap.add_argument("--sampler", default="dpmpp_2m")
    ap.add_argument("--scheduler", default="karras")
    ap.add_argument("--ckpt", default="Realistic_Vision_V5.1_fp16-no-ema.safetensors")
    ap.add_argument("--upscale-model", default=None,
                    help="如 RealESRGAN_x4.pth（VAEDecode 后接 4x 放大）")
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    w, h = (int(x) for x in args.size.lower().split("x"))
    p = dict(prompt=args.prompt, negative=args.negative, w=w, h=h, seed=args.seed,
             steps=args.steps, cfg=args.cfg, sampler=args.sampler,
             scheduler=args.scheduler, denoise=args.denoise)
    cfg = dict(ckpt=args.ckpt, cn_end=args.cn_end, upscale_model=args.upscale_model)

    ctrls = []
    if args.depth:
        ctrls.append((stage(args.depth, "depth"), "control_v11f1p_sd15_depth.pth", args.cn_depth))
    if args.lineart:
        ctrls.append((stage(args.lineart, "lineart"), "control_v11p_sd15_lineart.pth", args.cn_lineart))
    if args.normal:
        ctrls.append((stage(args.normal, "normal"), "control_v11p_sd15_normalbae.pth", args.cn_normal))
    if not ctrls:
        sys.exit("至少给 --depth / --lineart / --normal 之一")

    wf, out_id = build(stage(args.beauty, "beauty"), ctrls, p, cfg)
    t0 = time.time()
    try:
        r = http_json("/prompt", {"prompt": wf, "client_id": uuid.uuid4().hex})
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        sys.exit(f"提交失败 {e.code}: {body[:1200]}")
    pid = r.get("prompt_id")
    print(f"queued {pid}  ({w}x{h}, denoise={args.denoise}, ctrls={len(ctrls)})", flush=True)

    deadline = time.time() + args.timeout
    while time.time() < deadline:
        time.sleep(2)
        try:
            hist = http_json(f"/history/{pid}")
        except Exception:
            continue
        if pid not in hist:
            continue
        outs = hist[pid].get("outputs", {}).get(out_id, {}).get("images", [])
        if not outs:
            status = hist[pid].get("status", {})
            sys.exit(f"完成但无输出: {json.dumps(status)[:600]}")
        im = outs[0]
        q = urllib.parse.urlencode({"filename": im["filename"],
                                    "subfolder": im.get("subfolder", ""),
                                    "type": im.get("type", "output")})
        with urllib.request.urlopen(f"{HOST}/view?{q}", timeout=60) as r:
            data = r.read()
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "wb") as f:
            f.write(data)
        print(f"OK {args.out}  {len(data)//1024}KB  {time.time()-t0:.1f}s", flush=True)
        return
    sys.exit(f"超时 {args.timeout}s（prompt_id={pid}）")


if __name__ == "__main__":
    main()
