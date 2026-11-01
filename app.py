#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Anima LoRA Trainer - 本地训练 WebUI
零第三方依赖（标准库 http.server），复用 D:/anima-train 已跑通的训练链路。
"""
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs, quote
import urllib.request
import io

def _env_path(name, default):
    v = os.environ.get(name)
    return Path(v) if v else Path(default)


# 所有路径都可以用环境变量覆盖，见 README「配置」
BASE = _env_path("ANIMA_BASE", r"D:/anima-train")
VENV_PY = BASE / "venv" / "Scripts" / "python.exe"
ACCEL = BASE / "venv" / "Scripts" / "accelerate.exe"
SDS = BASE / "sd-scripts"
MODELS = BASE / "models"
OUTPUT = BASE / "output"
RUNS = BASE / "runs"
INFER = BASE / "infer" / "webui"
STATIC = Path(__file__).parent / "static"
UPLOADS = BASE / "uploads"

# 可选：单图生成素材所用的 ComfyUI 插件与 ComfyUI 本体
KIT = _env_path("LDK_KIT", r"D:/w/2026-10-30-01-37-50/comfyui-lora-dataset-kit")
COMFY = _env_path("COMFY_ROOT", r"D:/Comfy-Desktop/ComfyUI-Installs/Minimax H3 低显存专用/ComfyUI")
COMFY_IN = COMFY / "input"
COMFY_OUT = COMFY / "output"
COMFY_OUT_ALT = _env_path("COMFY_OUTPUT_DIR", r"D:/Comfy-Desktop/ComfyUI-Shared/output")
COMFY_IN_ALT = _env_path("COMFY_INPUT_DIR", r"D:/Comfy-Desktop/ComfyUI-Shared/input")
PRESETS = {"smoke_4": 4, "ring_9": 9, "face_18": 18, "standard_20": 20,
           "extended_30": 30, "matrix_54": 54}
PORT = 7861

DIT = MODELS / "anima-base-v1.0.safetensors"
TE = MODELS / "qwen_3_06b_base.safetensors"
VAE = MODELS / "qwen_image_vae.safetensors"
T5TOK = SDS / "configs" / "t5_old"

IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

STATE = {
    "job": None,            # "train" | "infer"
    "status": "idle",       # idle | running | done | failed | stopped
    "started_at": None,
    "ended_at": None,
    "progress": {"cur": 0, "total": 0, "speed": "", "loss": "", "elapsed": "", "eta": ""},
    "log": deque(maxlen=3000),
    "result": None,
    "config": None,
    "run_name": None,
}
LOCK = threading.Lock()
PROC = {"p": None}
CANCEL = threading.Event()


def now():
    return datetime.now().strftime("%H:%M:%S")


def log(line):
    with LOCK:
        STATE["log"].append(line)


def read_stream(pipe):
    """tqdm 用 \\r 刷新，必须同时按 \\r 和 \\n 切分。"""
    buf = ""
    while True:
        chunk = pipe.read(4096)
        if not chunk:
            break
        if isinstance(chunk, bytes):
            chunk = chunk.decode("utf-8", "replace")
        buf += chunk
        parts = re.split(r"[\r\n]", buf)
        buf = parts.pop()
        for p in parts:
            p = p.replace("\x1b[?25l", "").replace("\x1b[?25h", "").strip()
            if not p:
                continue
            log(p)
            parse_progress(p)
    if buf.strip():
        log(buf.strip())
        parse_progress(buf.strip())


PROG_RE = re.compile(r"(\d+)/(\d+)\s*\[([0-9:]+)<([0-9:]+),\s*([0-9.]+)\s*(s/it|it/s)")
LOSS_RE = re.compile(r"avr_loss=([0-9.]+)")
GEN_RE = re.compile(r"^\s*(\d+)\s*张\s+(\d+)s")


def parse_progress(line):
    m = PROG_RE.search(line)
    if m:
        cur, total, elapsed, eta, spd, unit = m.groups()
        if int(total) > 0 and int(total) != STATE["progress"]["total"]:
            pass
        STATE["progress"].update({
            "cur": int(cur), "total": int(total), "elapsed": elapsed,
            "eta": eta, "speed": f"{spd} {unit}",
        })
    m3 = GEN_RE.match(line)
    if m3:
        STATE["progress"].update({"cur": int(m3.group(1)), "total": PRESETS.get(
            (STATE.get("config") or {}).get("preset", "standard_20"), 20), "elapsed": m3.group(2) + "s"})
    m2 = LOSS_RE.search(line)
    if m2:
        STATE["progress"]["loss"] = m2.group(1)


def run_job(cmd, kind, cwd=None):
    try:
        p = subprocess.Popen(
            cmd, cwd=str(cwd or SDS), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, bufsize=0,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )
        PROC["p"] = p
        t = threading.Thread(target=read_stream, args=(p.stdout,), daemon=True)
        t.start()
        rc = p.wait()
        t.join(timeout=5)
        with LOCK:
            STATE["ended_at"] = time.time()
            if STATE.get("status") != "paused":
                STATE["status"] = "done" if rc == 0 else ("stopped" if rc in (1, -9, 3221225786) else "failed")
                if rc == 0 and STATE.get("progress", {}).get("total"):
                    STATE["progress"]["cur"] = STATE["progress"]["total"]
        log(f"[{now()}] 进程退出，返回码 {rc}")
    except Exception as e:
        with LOCK:
            STATE["status"] = "failed"
        log(f"[{now()}] 启动失败: {e}")
    finally:
        PROC["p"] = None


def gpu_stats():
    out = {"gpu_mem": None, "gpu_util": None, "gpu_power": None, "gpu_temp": None, "ram_used": None, "ram_total": None}
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=6)
        v = [x.strip() for x in r.stdout.strip().split(",")]
        out.update(gpu_mem=int(float(v[0])), gpu_total=int(float(v[1])), gpu_util=int(float(v[2])),
                   gpu_power=round(float(v[3])), gpu_temp=int(float(v[4])))
    except Exception:
        pass
    try:
        import psutil
        vm = psutil.virtual_memory()
        out["ram_used"] = round(vm.used / 1e9, 1)
        out["ram_total"] = round(vm.total / 1e9, 1)
    except Exception:
        pass
    return out


def _is_comfy(pid):
    try:
        import psutil
        cl = " ".join(psutil.Process(int(pid)).cmdline())
        return "ComfyUI" in cl
    except Exception:
        return False


def gpu_holders():
    out = []
    try:
        r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name",
                            "--format=csv,noheader"], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=8)
        for line in r.stdout.strip().splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 2 and parts[0].isdigit():
                name = Path(parts[1]).name
                if name.lower() in ("python.exe", "pythonw.exe"):
                    name = "ComfyUI" if _is_comfy(parts[0]) else "Python"
                out.append({"pid": int(parts[0]), "name": name})
    except Exception:
        pass
    return out


def comfy_output_root():
    """Comfy Desktop 会把 output 重定向到共享目录，取最近被写入的那个。"""
    best, best_t = None, -1.0
    for c in (COMFY_OUT, COMFY_OUT_ALT):
        if not c.is_dir():
            continue
        try:
            t = max((p.stat().st_mtime for p in c.iterdir()), default=0.0)
        except Exception:
            t = 0.0
        if t > best_t:
            best, best_t = c, t
    return best or COMFY_OUT


def comfy_input_root():
    """input 与 output 通常同属一个共享根，优先取与 output 同级的那一个。"""
    sibling = comfy_output_root().parent / "input"
    if sibling.is_dir():
        return sibling
    for c in (COMFY_IN_ALT, COMFY_IN):
        if c.is_dir():
            return c
    return COMFY_IN


def find_generated(name):
    for root in (comfy_output_root(), COMFY_OUT_ALT, COMFY_OUT):
        cand = root / "lora_dataset" / name
        if cand.is_dir():
            return cand
    return None


def comfy_online():
    try:
        with urllib.request.urlopen("http://127.0.0.1:8188/system_stats", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def parse_multipart(body, ctype):
    m = re.search(r"boundary=(?:\"([^\"]+)\"|([^;]+))", ctype or "")
    if not m:
        return None, None
    b = (m.group(1) or m.group(2)).strip().encode()
    for part in body.split(b"--" + b):
        if b"\r\n\r\n" not in part:
            continue
        head, data = part.split(b"\r\n\r\n", 1)
        fm = re.search(rb'filename="([^"]*)"', head)
        if not fm:
            continue
        if data.endswith(b"\r\n"):
            data = data[:-2]
        return fm.group(1).decode("utf-8", "replace"), data
    return None, None


def genmat(cfg):
    ref = Path(cfg["image"])
    if not ref.exists():
        raise RuntimeError("参考图不存在")
    if not comfy_online():
        raise RuntimeError("ComfyUI 未运行（127.0.0.1:8188 无响应），请先启动 ComfyUI")
    name = re.sub(r"[^A-Za-z0-9_\-]+", "_", cfg.get("name") or "character_new").strip("_") or "character_new"
    preset = cfg.get("preset") or "standard_20"
    total = PRESETS.get(preset, 20)
    rel = f"lora_dataset/{name}"
    dst = Path(cfg.get("dst") or (BASE / "dataset" / name))
    seed = int(cfg.get("seed", 20261101))
    cmd = [
        str(VENV_PY), "-u", str(KIT / "tools" / "run_dataset.py"),
        "--ref", str(ref), "--preset", preset,
        "--trigger", cfg.get("trigger") or "sks person",
        "--out", rel, "--resolution", str(int(cfg.get("resolution", 1024))),
        "--seed", str(seed),
        "--input-dir", str(comfy_input_root()), "--output-dir", str(comfy_output_root()),
    ]
    if int(cfg.get("max_shots") or 0) > 0:
        cmd += ["--max-shots", str(int(cfg["max_shots"]))]
    if cfg.get("resume"):
        cmd += ["--resume"]
    log(f"[生成] 预设 {preset}（{total} 张）| 种子 {seed} | 输出 {comfy_output_root() / 'lora_dataset' / name}")
    log(f"[生成] 参考图 {ref.name}")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    p = subprocess.Popen(cmd, cwd=str(KIT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         bufsize=0, env=env)
    PROC["p"] = p
    threading.Thread(target=read_stream, args=(p.stdout,), daemon=True).start()
    rc = p.wait()
    PROC["p"] = None
    if rc != 0:
        raise RuntimeError(f"生成进程返回码 {rc}（详见日志）")
    src_dir = find_generated(name)
    if src_dir is None:
        raise RuntimeError(f"未找到输出目录（已查 {COMFY_OUT} 与 {COMFY_OUT_ALT}）")
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in src_dir.iterdir():
        if f.is_file():
            shutil.copy2(f, dst / f.name)
            if f.suffix.lower() in (".png", ".txt"):
                n += 1
    log(f"[生成] 完成：复制 {n} 个文件 → {dst}")
    STATE["result"] = str(dst)


def preflight_gpu(need_mb=1500):
    st = gpu_stats()
    used = st.get("gpu_mem") or 0
    if used > need_mb:
        who = "、".join(f"{h['name']}(PID {h['pid']})" for h in gpu_holders()) or "未知进程"
        return (f"显存已被占用 {used} MiB，本任务需要几乎全部 8GB。占用者：{who}。"
                f"请先关闭，或点「释放显存」。")
    return None


def release_gpu():
    killed = []
    for h in gpu_holders():
        if h["name"] == "ComfyUI":
            try:
                subprocess.run(["taskkill", "/PID", str(h["pid"]), "/T", "/F"],
                               capture_output=True, timeout=15)
                killed.append(h["pid"])
            except Exception:
                pass
    return killed


def _parse_cmd(txt):
    import re as _re
    out = {}
    toks = [a or b for a, b in _re.findall(r'"([^"]*)"|(\S+)', txt)]
    for i, t in enumerate(toks):
        if t.startswith("--") and i + 1 < len(toks):
            out[t[2:]] = toks[i + 1]
    return toks, out


def _state_step(p):
    m = re.search(r"-step(\d+)-state$", Path(str(p)).name)
    return int(m.group(1)) if m else 0


def find_state(name):
    if not OUTPUT.is_dir():
        return None
    cands = [d for d in OUTPUT.glob(f"{name}*state") if d.is_dir()]
    if not cands:
        return None
    return max(cands, key=lambda d: d.stat().st_mtime)


def list_runs():
    recs = []
    if not RUNS.is_dir():
        return recs
    for d in sorted(RUNS.iterdir(), key=lambda x: -x.stat().st_mtime):
        if not d.is_dir():
            continue
        cmd_file, meta_file = d / "command.txt", d / "meta.json"
        if not (cmd_file.exists() or meta_file.exists()):
            continue
        meta = {}
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        args = {}
        if cmd_file.exists():
            try:
                _, args = _parse_cmd(cmd_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        dataset = meta.get("dataset") or ""
        if not dataset:
            toml = d / "dataset.toml"
            if toml.exists():
                m = re.search(r'image_dir\s*=\s*"([^"]+)"', toml.read_text(encoding="utf-8"))
                dataset = m.group(1) if m else ""
        resolution = meta.get("resolution") or ""
        if not resolution:
            toml_f = d / "dataset.toml"
            if toml_f.exists():
                m2 = re.search(r"resolution\s*=\s*(\d+)", toml_f.read_text(encoding="utf-8"))
                resolution = m2.group(1) if m2 else ""
        n_img = 0
        if dataset and Path(dataset).is_dir():
            n_img = sum(1 for f in Path(dataset).iterdir() if f.suffix.lower() in IMG_EXT)
        steps_cfg = int(meta.get("steps") or args.get("max_train_steps") or 0)
        out_file = OUTPUT / f"{d.name}.safetensors"
        loss_info = {}
        tb = d / "tb"
        if tb.is_dir():
            r = _read_one(tb, d.name)
            if r.get("ok"):
                loss_info = {"last_step": r["last_step"], "last_loss": r["last_loss"],
                             "min_loss": r["min_loss"], "points": r["count"]}
        started = meta.get("started_at") or datetime.fromtimestamp(d.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        dur = ""
        if out_file.exists() and meta.get("started_at"):
            try:
                t0 = datetime.strptime(meta["started_at"], "%Y-%m-%d %H:%M:%S")
                dur = int((datetime.fromtimestamp(out_file.stat().st_mtime) - t0).total_seconds())
            except Exception:
                dur = ""
        recs.append({
            "name": d.name, "started_at": started, "dataset": Path(dataset).name if dataset else "—",
            "dataset_path": dataset, "images": n_img,
            "steps_cfg": steps_cfg, "dim": meta.get("dim") or args.get("network_dim") or "",
            "lr": meta.get("lr") or args.get("learning_rate") or "",
            "optimizer": meta.get("optimizer") or args.get("optimizer_type") or "",
            "seed": meta.get("seed") or args.get("seed") or "",
            "resolution": resolution,
            "bundle": args.get("network_module", "") == "networks.lora_anima",
            "output": {"exists": out_file.exists(),
                       "mb": round(out_file.stat().st_size / 1e6, 1) if out_file.exists() else 0,
                       "path": str(out_file) if out_file.exists() else ""},
            "loss": loss_info,
            "duration_s": dur,
            "state": (str(find_state(d.name)) if find_state(d.name) else ""),
            "total_steps": steps_cfg,
            "status": "完成" if out_file.exists() else ("已训练·无产物" if loss_info else "仅配置"),
        })
    known = {r["name"] for r in recs}
    if OUTPUT.is_dir():
        for f in OUTPUT.glob("*.safetensors"):
            if "-step" in f.name or f.stem in known:
                continue
            st = f.stat()
            recs.append({
                "name": f.stem, "started_at": datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M"),
                "dataset": "—", "dataset_path": "", "images": 0, "steps_cfg": 0,
                "dim": "", "lr": "", "optimizer": "", "seed": "", "resolution": "",
                "bundle": True,
                "output": {"exists": True, "mb": round(st.st_size / 1e6, 1), "path": str(f)},
                "loss": {}, "duration_s": "", "status": "完成·无参数记录",
            })
    recs.sort(key=lambda r: r["started_at"], reverse=True)
    return recs


def _run_dirs():
    out = []
    if RUNS.is_dir():
        for d in sorted(RUNS.iterdir(), key=lambda x: -x.stat().st_mtime):
            if (d / "tb").is_dir():
                out.append(d)
    return out


def _read_one(base, run_name, max_points=700):
    if not base or not base.is_dir():
        return {"ok": False, "error": "没有找到日志目录"}
    files = sorted(base.rglob("events.out.tfevents*"), key=lambda x: x.stat().st_mtime)
    if not files:
        return {"ok": False, "error": "该次训练还没有写出日志（启动后约需 10-30 秒）"}
    dirs, seen = [], set()
    for f in files:
        if f.parent not in seen:
            seen.add(f.parent)
            dirs.append(f.parent)
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        pts, avg, cum = [], [], 0
        tag_used = None
        for idx, d in enumerate(dirs):
            ea = EventAccumulator(str(d), size_guidance={"scalars": 0})
            ea.Reload()
            tags = ea.Tags().get("scalars", [])
            tag = next((t for t in ("loss/current", "loss", "train/loss") if t in tags), None)
            if tag is None:
                continue
            tag_used = tag
            off = cum
            block = [(int(e.step) + off, float(e.value)) for e in ea.Scalars(tag)]
            if block:
                pts += block
                cum = block[-1][0]
            if "loss/average" in tags:
                avg += [(int(e.step) + off, float(e.value)) for e in ea.Scalars("loss/average")]
        if not pts:
            return {"ok": False, "error": "日志中暂无 loss 标量"}
    except Exception as e:
        return {"ok": False, "error": f"读取失败: {e}"}
    if not pts:
        return {"ok": False, "error": "日志为空"}
    total = len(pts)
    if total > max_points:
        stride = total // max_points + 1
        pts = pts[::stride]
    if avg and len(avg) > max_points:
        stride = len(avg) // max_points + 1
        avg = avg[::stride]
    return {
        "ok": True, "tag": tag, "run": run_name,
        "steps": [p[0] for p in pts], "loss": [round(p[1], 5) for p in pts],
        "avg_steps": [p[0] for p in (avg or [])], "avg": [round(p[1], 5) for p in (avg or [])],
        "count": total, "shown": len(pts),
        "last_step": pts[-1][0], "last_loss": round(pts[-1][1], 4),
        "min_loss": round(min(p[1] for p in pts), 4),
    }


def read_loss(run_name=None, max_points=700):
    runs = [d.name for d in _run_dirs()]
    if run_name:
        r = _read_one(RUNS / run_name / "tb", run_name, max_points)
        r["runs"] = runs
        return r
    last = None
    for d in _run_dirs():
        r = _read_one(d / "tb", d.name, max_points)
        if r.get("ok"):
            r["runs"] = runs
            return r
        last = r
    out = last or {"ok": False, "error": "还没有任何训练日志"}
    out["runs"] = runs
    return out


def scan_dataset(path):
    p = Path(path)
    if not p.is_dir():
        return {"ok": False, "error": "目录不存在"}
    imgs = [f for f in p.iterdir() if f.suffix.lower() in IMG_EXT]
    caps = [f for f in p.iterdir() if f.suffix.lower() == ".txt"]
    stems = {f.stem for f in imgs}
    missing = [f.name for f in imgs if f.stem not in {c.stem for c in caps}]
    samples = []
    for f in sorted(caps)[:3]:
        try:
            samples.append({"name": f.name, "text": f.read_text(encoding="utf-8", errors="replace")[:300]})
        except Exception:
            pass
    quality_words = 0
    for f in caps:
        try:
            t = f.read_text(encoding="utf-8", errors="replace").lower()
        except Exception:
            continue
        if any(w in t for w in ("masterpiece", "best quality", "8k", "ultra detailed", "photorealistic")):
            quality_words += 1
    sizes = []
    try:
        from PIL import Image
        for f in imgs[:12]:
            with Image.open(f) as im:
                sizes.append(im.size)
    except Exception:
        pass
    return {
        "ok": True, "images": len(imgs), "captions": len(caps),
        "missing_caption": len(missing), "quality_word_files": quality_words,
        "samples": samples, "sizes": sorted({f"{w}x{h}" for w, h in sizes}),
        "has_trigger_hint": True,
    }


def list_outputs():
    if not OUTPUT.exists():
        return []
    items = []
    for f in sorted(OUTPUT.glob("*.safetensors"), key=lambda x: -x.stat().st_mtime):
        st = f.stat()
        items.append({"name": f.name, "path": str(f), "mb": round(st.st_size / 1e6, 1),
                      "mtime": datetime.fromtimestamp(st.st_mtime).strftime("%m-%d %H:%M")})
    return items


def build_train_cmd(cfg):
    name = cfg.get("name") or f"anima_{datetime.now():%m%d_%H%M}"
    run_dir = RUNS / name
    run_dir.mkdir(parents=True, exist_ok=True)
    res = int(cfg.get("resolution", 640))
    toml = f"""[general]
caption_extension = ".txt"
shuffle_caption = false

[[datasets]]
resolution = {res}
batch_size = {int(cfg.get("batch_size", 1))}
enable_bucket = true
bucket_no_upscale = false

[[datasets.subsets]]
image_dir = "{cfg['dataset'].replace(chr(92), '/')}"
num_repeats = 1
"""
    (run_dir / "dataset.toml").write_text(toml, encoding="utf-8")
    steps = int(cfg.get("steps", 300))
    save_every = int(cfg.get("save_every") or max(50, min(200, steps // 8)))
    dim = int(cfg.get("dim", 32))
    lr = str(cfg.get("lr", "2e-5"))
    optimizer = cfg.get("optimizer", "AdamW8bit")
    seed = int(cfg.get("seed", 42))
    cmd = [
        str(ACCEL), "launch", "--num_processes", "1", "--num_cpu_threads_per_process", "1",
        "--mixed_precision", "bf16",
        "anima_train_network.py",
        "--pretrained_model_name_or_path", str(DIT),
        "--qwen3", str(TE),
        "--vae", str(VAE),
        "--t5_tokenizer_path", str(T5TOK),
        "--dataset_config", str(run_dir / "dataset.toml"),
        "--output_dir", str(OUTPUT),
        "--output_name", name,
        "--save_model_as", "safetensors",
        "--network_module", "networks.lora_anima",
        "--network_dim", str(dim), "--network_alpha", str(dim),
        "--network_train_unet_only",
        "--learning_rate", lr,
        "--optimizer_type", optimizer,
        "--lr_scheduler", "constant",
        "--max_train_steps", str(steps),
        "--save_every_n_steps", str(save_every),
        "--mixed_precision", "bf16",
        "--gradient_checkpointing",
        "--cache_latents",
        "--cache_text_encoder_outputs",
        "--max_data_loader_n_workers", "2",
        "--persistent_data_loader_workers",
        "--vae_chunk_size", "64",
        "--vae_disable_cache",
        "--qwen_image_vae_2d",
        "--timestep_sampling", "sigmoid",
        "--seed", str(seed),
        "--logging_dir", str(run_dir / "tb"),
        "--save_state",
    ]
    resume_state = cfg.get("resume_state")
    if resume_state:
        cmd += ["--resume", str(resume_state)]
        log(f"[{now()}] 从状态续训：{Path(resume_state).name}")
    (run_dir / "command.txt").write_text(" ".join(f'"{c}"' if " " in c else c for c in cmd), encoding="utf-8")
    import json as _json
    (run_dir / "meta.json").write_text(_json.dumps({
        "name": name, "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": cfg, "steps": int(cfg.get("total_steps") or steps), "session_steps": steps,
        "dim": dim, "lr": lr, "optimizer": optimizer,
        "seed": seed, "resolution": res, "dataset": cfg["dataset"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return cmd, name


def build_infer_cmd(cfg):
    INFER.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(VENV_PY), "anima_minimal_inference.py",
        "--dit", str(DIT), "--text_encoder", str(TE), "--vae", str(VAE),
        "--qwen_image_vae_2d", "--vae_chunk_size", "64",
        "--image_size", str(int(cfg.get("width", 512))), str(int(cfg.get("height", 768))),
        "--infer_steps", str(int(cfg.get("steps", 20))),
        "--guidance_scale", str(cfg.get("guidance", 5)),
        "--seed", str(int(cfg.get("seed", 42))),
        "--prompt", cfg.get("prompt", "1girl, masterpiece, best quality"),
        "--save_path", str(INFER),
    ]
    if cfg.get("negative"):
        cmd += ["--negative_prompt", cfg["negative"]]
    if cfg.get("lora"):
        cmd += ["--lora_weight", cfg["lora"], "--lora_multiplier", str(cfg.get("multiplier", 1.0))]
    return cmd


UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
CAP_PROMPTS = {
    "portrait": ("Caption this image as if you were going to generate it. Describe only pose, "
                 "clothing, scene, lighting and composition. Do not describe the person's face, "
                 "hair color or body shape. No preamble."),
    "general": ("Caption this image as if you were going to try to generate it with an image "
                "generator. Be thorough and describe everything in the image. Be decisive. "
                "No preamble."),
    "style": ("Describe the art style, medium, line quality, color palette and rendering "
              "technique of this image. Do not describe the subject's identity, face or hair. "
              "No preamble."),
}


def http_get(url, timeout=20, referer=None):
    h = {"User-Agent": UA}
    if referer:
        h["Referer"] = referer
    req = urllib.request.Request(url, headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def save_bytes(data, dst: Path, stem: str, min_side=0):
    from PIL import Image
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
    except Exception:
        return None
    if min_side and min(im.size) < min_side:
        return None
    ext = ".png" if (im.format or "").upper() == "PNG" else ".jpg"
    p = dst / f"{stem}{ext}"
    n = 1
    while p.exists():
        p = dst / f"{stem}_{n}{ext}"
        n += 1
    p.write_bytes(data)
    return p


def phash(im):
    import numpy as np
    from PIL import Image
    g = im.convert("L").resize((32, 32), Image.LANCZOS)
    a = np.asarray(g, dtype=np.float32)
    n = 8
    k = np.arange(32)[:, None]
    M = np.cos(np.pi * (2 * k + 1) * np.arange(n)[None, :] / 64.0) * np.sqrt(2.0 / 32)
    M[0, :] *= 1 / np.sqrt(2)
    d = (M.T @ a @ M)[:8, :8]
    med = np.median(d.flatten()[1:])
    v = 0
    for b in (d > med).flatten():
        v = (v << 1) | int(b)
    return v


def hamming(a, b):
    return bin(a ^ b).count("1")


def scrape_bing(cfg):
    dst = Path(cfg["dst"])
    dst.mkdir(parents=True, exist_ok=True)
    q = cfg["query"]
    want = int(cfg.get("count", 30))
    min_side = int(cfg.get("min_side", 512))
    urls, first = [], 0
    log(f"[采集] 关键词：{q}，目标 {want} 张，最小边 {min_side}px")
    while len(urls) < want * 3 and first < 400:
        u = (f"https://cn.bing.com/images/async?q={quote(q)}&first={first}"
             f"&count=35&mkt=zh-CN&adlt=off")
        try:
            html = http_get(u).decode("utf-8", "ignore")
        except Exception as e:
            log(f"[采集] 搜索请求失败：{e}")
            break
        found = [x.replace("\\/", "/") for x in re.findall(r'murl&quot;:&quot;(.*?)&quot;', html)]
        if not found:
            break
        for f in found:
            if f not in urls:
                urls.append(f)
        first += 35
        time.sleep(0.4)
    log(f"[采集] 候选 {len(urls)} 个，开始下载（失败的会被跳过）")
    ok = 0
    for i, u in enumerate(urls):
        if CANCEL.is_set():
            log("[采集] 已取消")
            break
        if ok >= want:
            break
        STATE["progress"].update({"cur": i + 1, "total": min(len(urls), want * 3)})
        try:
            p = save_bytes(http_get(u, timeout=15), dst, f"{cfg.get('prefix') or 'img'}_{ok + 1:03d}", min_side)
            if p:
                ok += 1
                if ok % 10 == 0:
                    log(f"[采集] 已保存 {ok} 张")
        except Exception:
            continue
    log(f"[采集] 完成：成功 {ok} 张 → {dst}")
    STATE["result"] = str(dst)


def download_urls(cfg):
    dst = Path(cfg["dst"])
    dst.mkdir(parents=True, exist_ok=True)
    urls = [x.strip() for x in (cfg.get("urls") or "").splitlines() if x.strip().startswith("http")]
    log(f"[采集] 直链 {len(urls)} 条")
    ok = 0
    for i, u in enumerate(urls):
        if CANCEL.is_set():
            log("[采集] 已取消")
            break
        STATE["progress"].update({"cur": i + 1, "total": len(urls)})
        try:
            p = save_bytes(http_get(u, timeout=20), dst, f"{cfg.get('prefix') or 'img'}_{ok + 1:03d}",
                           int(cfg.get("min_side", 0)))
            if p:
                ok += 1
        except Exception as e:
            log(f"[采集] 失败 {u[:60]}… {e}")
    log(f"[采集] 完成：成功 {ok}/{len(urls)} → {dst}")


def import_local(cfg):
    src, dst = Path(cfg["src"]), Path(cfg["dst"])
    dst.mkdir(parents=True, exist_ok=True)
    files = [f for f in src.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXT]
    log(f"[导入] 源目录 {len(files)} 张图")
    n = 0
    for f in files:
        target = dst / f.name
        k = 1
        while target.exists():
            target = dst / f"{f.stem}_{k}{f.suffix}"
            k += 1
        shutil.copy2(f, target)
        n += 1
    log(f"[导入] 完成：复制 {n} 张 → {dst}")


def process_images(cfg):
    import numpy as np
    import cv2
    from PIL import Image
    src, dst = Path(cfg["src"]), Path(cfg["dst"])
    dst.mkdir(parents=True, exist_ok=True)
    target = int(cfg.get("size", 1024))
    blur_th = float(cfg.get("blur_th", 60))
    dup_th = int(cfg.get("dup_th", 6))
    crop_face = bool(cfg.get("crop_face", True))
    white_bg = bool(cfg.get("white_bg", True))
    files = sorted(f for f in src.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXT)
    log(f"[处理] 输入 {len(files)} 张 | 目标长边 {target} | 清晰度阈值 {blur_th} | 去重阈值 {dup_th}")
    cascade = None
    if crop_face:
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    kept, noface, blur, dup, err = 0, 0, 0, 0, 0
    hashes = []
    for i, f in enumerate(files):
        if CANCEL.is_set():
            log("[处理] 已取消")
            break
        STATE["progress"].update({"cur": i + 1, "total": len(files)})
        try:
            im = Image.open(f)
            im.load()
            if im.mode in ("RGBA", "LA", "P"):
                im = im.convert("RGBA")
                bg = Image.new("RGB", im.size, (255, 255, 255))
                bg.paste(im, mask=im.split()[-1])
                im = bg
            else:
                im = im.convert("RGB")
            arr = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
            if cascade is not None and not cascade.empty():
                faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(48, 48))
                if len(faces) == 0:
                    noface += 1
                    log(f"[处理] 未检测到人脸（二次元图属正常），保留原图: {f.name}")
                else:
                    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
                    cx, cy = x + w / 2, y + h / 2
                    bh = h * 2.6
                    bw = bh * 0.75
                    left = int(max(0, cx - bw / 2))
                    top = int(max(0, cy - bh * 0.42))
                    right = int(min(arr.shape[1], left + bw))
                    bottom = int(min(arr.shape[0], top + bh))
                    if right - left >= 64 and bottom - top >= 64:
                        arr = arr[top:bottom, left:right]
                        gray = gray[top:bottom, left:right]
            sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            if sharp < blur_th:
                blur += 1
                log(f"[处理] 跳过（模糊 {sharp:.0f}）: {f.name}")
                continue
            ph = phash(Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)))
            if any(hamming(ph, h0) <= dup_th for h0 in hashes):
                dup += 1
                continue
            hashes.append(ph)
            im2 = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
            w0, h0 = im2.size
            scale = target / max(w0, h0)
            if scale < 1:
                im2 = im2.resize((max(1, int(w0 * scale)), max(1, int(h0 * scale))), Image.LANCZOS)
            out = dst / f"{f.stem}.png"
            k = 1
            while out.exists():
                out = dst / f"{f.stem}_{k}.png"
                k += 1
            im2.save(out, "PNG")
            kept += 1
        except Exception as e:
            err += 1
            log(f"[处理] 出错 {f.name}: {e}")
    log(f"[处理] 完成：保留 {kept} | 无人脸 {noface} | 模糊 {blur} | 重复 {dup} | 出错 {err}")
    log(f"[处理] 输出目录：{dst}")
    STATE["result"] = str(dst)


def run_caption(cfg):
    target = Path(cfg["dir"])
    if not target.is_dir():
        raise RuntimeError("目录不存在")
    mode = cfg.get("mode", "portrait")
    prompt = CAP_PROMPTS.get(mode, CAP_PROMPTS["portrait"])
    yaml_path = RUNS / f"caption_{datetime.now():%m%d_%H%M%S}.yaml"
    RUNS.mkdir(parents=True, exist_ok=True)
    model = cfg.get("model") or "Qwen/Qwen3-VL-4B-Instruct"
    text = (
        "job: extension\n"
        "config:\n"
        '  name: "caption_job"\n'
        "  process:\n"
        "    - type: 'Qwen3VLCaptioner'\n"
        f"      device: '{cfg.get('device', 'cuda')}'\n"
        "      caption:\n"
        f"        model_name_or_path: '{model}'\n"
        "        dtype: bf16\n"
        "        quantize: true\n"
        "        qtype: 'convrot8'\n"
        "        low_vram: true\n"
        "        extensions: [ 'png', 'jpg', 'jpeg', 'webp' ]\n"
        f"        path_to_caption: '{target.as_posix()}'\n"
        f"        recaption: {'true' if cfg.get('recaption') else 'false'}\n"
        "        compile: false\n"
        "        caption_extension: 'txt'\n"
        "        caption_prompt: >-\n"
        f"          {prompt}\n"
    )
    yaml_path.write_text(text, encoding="utf-8")
    log(f"[打标] 模式 {mode} | 模型 {model} | 目录 {target}")
    log(f"[打标] 首次加载模型约 1–2 分钟，之后约 15 秒/张")
    cmd = [str(Path(r"D:/ai-toolkit/.venv/Scripts/python.exe")), "run.py", str(yaml_path)]
    env = dict(os.environ, HF_HUB_OFFLINE="1", PYTHONIOENCODING="utf-8")
    p = subprocess.Popen(cmd, cwd=r"D:/ai-toolkit", stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, bufsize=0, env=env)
    PROC["p"] = p
    t = threading.Thread(target=read_stream, args=(p.stdout,), daemon=True)
    t.start()
    rc = p.wait()
    t.join(timeout=5)
    PROC["p"] = None
    if rc != 0:
        raise RuntimeError(f"打标进程返回码 {rc}")
    trigger = (cfg.get("trigger") or "").strip()
    n = 0
    for f in sorted(target.glob("*.txt")):
        txt = f.read_text(encoding="utf-8", errors="replace").strip()
        if not txt:
            continue
        n += 1
        if trigger and not txt.startswith(trigger):
            f.write_text(f"{trigger}, {txt}", encoding="utf-8")
    log(f"[打标] 完成：{n} 个标注文件" + (f"（已加触发词 {trigger}）" if trigger else ""))


def run_py_task(kind, fn, cfg):
    with LOCK:
        if STATE["status"] == "running":
            return {"ok": False, "error": "已有任务在运行"}
        STATE.update({
            "job": kind, "status": "running", "started_at": time.time(),
            "ended_at": None, "result": None, "config": cfg,
            "progress": {"cur": 0, "total": 0, "speed": "", "loss": "", "elapsed": "", "eta": ""},
        })
        STATE["log"].clear()
    CANCEL.clear()

    def wrapper():
        try:
            fn(cfg)
            with LOCK:
                STATE["status"] = "done"
                STATE["ended_at"] = time.time()
        except Exception as e:
            log(f"[错误] {e}")
            with LOCK:
                STATE["status"] = "failed"
                STATE["ended_at"] = time.time()

    threading.Thread(target=wrapper, daemon=True).start()
    return {"ok": True}


def start_job(kind, cfg):
    if kind in ("train", "caption"):
        err = preflight_gpu()
        if err:
            log(f"[预检] {err}")
            return {"ok": False, "error": err}
    TASKS = {"scrape": scrape_bing, "urls": download_urls, "import": import_local,
             "process": process_images, "caption": run_caption, "genmat": genmat}
    if kind in TASKS:
        return run_py_task(kind, TASKS[kind], cfg)
    with LOCK:
        if STATE["status"] == "running":
            return {"ok": False, "error": "已有任务在运行，请先停止或等待完成"}
        STATE.update({
            "job": kind, "status": "running", "started_at": time.time(),
            "ended_at": None, "result": None, "config": cfg,
            "progress": {"cur": 0, "total": 0, "speed": "", "loss": "", "elapsed": "", "eta": ""},
        })
        STATE["log"].clear()
    if kind == "train":
        try:
            cmd, name = build_train_cmd(cfg)
        except Exception as e:
            with LOCK:
                STATE["status"] = "failed"
            return {"ok": False, "error": f"配置有误: {e}"}
        STATE["run_name"] = name
        log(f"[{now()}] 训练启动: {name}")
        log(f"[{now()}] 数据集: {cfg['dataset']} | {cfg.get('steps')} 步 | rank {cfg.get('dim')} | {cfg.get('resolution')}px")
        threading.Thread(target=run_job, args=(cmd, kind), daemon=True).start()
        return {"ok": True, "name": name}
    else:
        before = {p.name for p in INFER.glob("*.png")}
        cmd = build_infer_cmd(cfg)
        log(f"[{now()}] 推理启动（约 40-60 秒）")
        threading.Thread(target=run_job, args=(cmd, kind), daemon=True).start()
        threading.Thread(target=collect_infer, args=(before,), daemon=True).start()
        return {"ok": True}


def collect_infer(before):
    """等待推理结束，取最新生成的 png。"""
    for _ in range(600):
        time.sleep(1)
        if STATE["status"] != "running":
            break
    newest = None
    for p in INFER.rglob("*.png"):
        if p.name not in before and p.stat().st_size > 0:
            if newest is None or p.stat().st_mtime > newest.stat().st_mtime:
                newest = p
    if newest:
        with LOCK:
            STATE["result"] = str(newest)
        log(f"[{now()}] 出图: {newest}")


def pause_job():
    p = PROC.get("p")
    if not p or STATE["status"] != "running" or STATE["job"] != "train":
        return {"ok": False, "error": "当前没有正在运行的训练"}
    name = STATE.get("run_name")
    at = (STATE.get("progress") or {}).get("cur") or 0
    try:
        subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True, timeout=20)
    except Exception as e:
        return {"ok": False, "error": str(e)}
    with LOCK:
        STATE["status"] = "paused"
        STATE["ended_at"] = time.time()
    st = find_state(name) if name else None
    if name:
        run_dir = RUNS / name
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "paused.json").write_text(json.dumps({
            "at_step": at, "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "state": str(st) if st else "",
        }, ensure_ascii=False), encoding="utf-8")
    msg = f"已暂停于第 {at} 步" + (f"，状态已保存：{st.name}" if st else "（尚未到达保存点）")
    log(f"[{now()}] {msg}")
    return {"ok": True, "at_step": at, "state": str(st) if st else "", "message": msg}


def resume_job(name):
    run_dir = RUNS / name
    meta_file = run_dir / "meta.json"
    if not meta_file.exists():
        return {"ok": False, "error": "找不到该任务的参数快照"}
    try:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
    except Exception as e:
        return {"ok": False, "error": f"参数快照损坏: {e}"}
    cfg = dict(meta.get("config") or {})
    if not cfg.get("dataset"):
        return {"ok": False, "error": "参数快照缺少数据集"}
    st = find_state(name)
    if st:
        cfg["resume_state"] = str(st)
    else:
        lora = OUTPUT / f"{name}.safetensors"
        if not lora.exists():
            return {"ok": False, "error": "既没有训练状态，也没有已保存的 LoRA，无法续训"}
    cfg["name"] = name
    total = int(meta.get("steps") or cfg.get("steps") or 0)
    done = _state_step(st) if st else 0
    if done and total:
        cfg["steps"] = max(1, total - done)
        cfg["total_steps"] = total
    else:
        cfg["steps"] = total
    rec = {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "offset": done,
           "state": st.name if st else "", "steps": cfg["steps"]}
    with open(run_dir / "sessions.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return start_job("train", cfg)


def stop_job():
    p = PROC.get("p")
    if not p:
        CANCEL.set()
        with LOCK:
            if STATE["status"] == "running":
                STATE["status"] = "stopped"
                STATE["ended_at"] = time.time()
        log(f"[{now()}] 已请求取消")
        return {"ok": True}
    try:
        subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True, timeout=15)
        with LOCK:
            STATE["status"] = "stopped"
            STATE["ended_at"] = time.time()
        log(f"[{now()}] 已手动停止")
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False)
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path in ("/", "/index.html"):
            return self._file(STATIC / "index.html", "text/html; charset=utf-8")
        if u.path.startswith("/static/"):
            return self._file(STATIC / u.path.split("/static/")[-1])
        if u.path == "/api/status":
            with LOCK:
                tail = list(STATE["log"])[-160:]
                payload = {
                    "job": STATE["job"], "status": STATE["status"],
                    "progress": STATE["progress"], "result": STATE["result"],
                    "log": tail, "config": STATE["config"],
                    "elapsed": round(time.time() - STATE["started_at"]) if STATE["started_at"] and STATE["status"] == "running" else None,
                }
            payload["stats"] = gpu_stats()
            return self._send(200, payload)
        if u.path == "/api/defaults":
            datasets = []
            ds_root = BASE / "dataset"
            if ds_root.is_dir():
                datasets = [str(p) for p in ds_root.iterdir() if p.is_dir()]
            return self._send(200, {
                "datasets": datasets, "outputs": list_outputs(),
                "paths": {"dit": str(DIT), "te": str(TE), "vae": str(VAE), "sd_scripts": str(SDS)},
                "exists": {"dit": DIT.exists(), "te": TE.exists(), "vae": VAE.exists(), "sds": SDS.exists()},
            })
        if u.path == "/api/gpu":
            return self._send(200, {"holders": gpu_holders(), "stats": gpu_stats()})
        if u.path == "/api/runs":
            return self._send(200, {"runs": list_runs()})
        if u.path == "/api/loss":
            return self._send(200, read_loss(q.get("run", [""])[0] or None))
        if u.path == "/api/comfy":
            return self._send(200, {"online": comfy_online(), "root": str(COMFY)})
        if u.path == "/api/outputs":
            return self._send(200, {"outputs": list_outputs()})
        if u.path == "/api/scan":
            return self._send(200, scan_dataset(q.get("path", [""])[0]))
        if u.path == "/api/image":
            p = Path(q.get("path", [""])[0])
            try:
                p.resolve().relative_to(BASE.resolve())
            except Exception:
                return self._send(403, {"error": "forbidden"})
            if p.suffix.lower() in IMG_EXT and p.exists():
                data = p.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            return self._send(404, {"error": "not found"})
        return self._send(404, {"error": "not found"})

    def _file(self, p, ctype=None):
        if not p.exists() or not p.is_file():
            return self._send(404, {"error": "not found"})
        ctype = ctype or {
            ".html": "text/html; charset=utf-8", ".js": "application/javascript; charset=utf-8",
            ".css": "text/css; charset=utf-8",
        }.get(p.suffix, "application/octet-stream")
        return self._send(200, p.read_bytes(), ctype)

    def do_POST(self):
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n) if n else b""
        if u.path == "/api/upload":
            fn, data = parse_multipart(raw, self.headers.get("Content-Type", ""))
            if not data:
                return self._send(400, {"ok": False, "error": "没有收到文件"})
            UPLOADS.mkdir(parents=True, exist_ok=True)
            ext = Path(fn or "").suffix.lower() or ".png"
            if ext not in IMG_EXT:
                return self._send(400, {"ok": False, "error": "仅支持图片格式"})
            out = UPLOADS / f"ref_{datetime.now():%m%d_%H%M%S}{ext}"
            out.write_bytes(data)
            return self._send(200, {"ok": True, "path": str(out), "name": fn})
        try:
            cfg = json.loads((raw or b"{}").decode("utf-8") or "{}")
        except Exception:
            return self._send(400, {"ok": False, "error": "JSON 解析失败"})
        if u.path == "/api/start":
            return self._send(200, start_job(cfg.get("kind", "train"), cfg))
        if u.path == "/api/release_gpu":
            return self._send(200, {"ok": True, "killed": release_gpu()})
        if u.path == "/api/pause":
            return self._send(200, pause_job())
        if u.path == "/api/resume":
            return self._send(200, resume_job(cfg.get("name", "")))
        if u.path == "/api/stop":
            return self._send(200, stop_job())
        if u.path == "/api/open":
            target = cfg.get("path", "")
            try:
                os.startfile(target)  # noqa
                return self._send(200, {"ok": True})
            except Exception as e:
                return self._send(200, {"ok": False, "error": str(e)})
        return self._send(404, {"error": "not found"})


def main():
    global PORT
    if not (SDS / "anima_train_network.py").exists():
        print("未找到 sd-scripts，请确认 D:/anima-train/sd-scripts 存在")
        input("按回车退出...")
        return
    srv = None
    for port in (PORT, PORT + 1, PORT + 2):
        try:
            srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
            PORT = port
            break
        except OSError:
            continue
    if srv is None:
        print("端口被占用，无法启动")
        input("按回车退出...")
        return
    url = f"http://127.0.0.1:{PORT}"
    print("=" * 58)
    print(f"  Anima LoRA Trainer 已启动: {url}")
    print("  关闭此窗口即停止服务")
    print("=" * 58)
    if not os.environ.get("ANIMA_NO_BROWSER"):
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
