#!/usr/bin/env python3
"""FastAPI dynamic-batching inference server for UTMOSv2 MOS prediction.

Predicts the naturalness MOS (Mean Opinion Score) of a speech clip:
  bytes -> 16 kHz mono -> UTMOSv2 features (3 s SSL waveform + 4 multi-res mels)
        -> fusion model (WavLM/SSL + EfficientNetV2) -> scalar MOS in ~[1, 5].

Throughput stack (single GPU; the bottleneck is the CPU feature stack, not the GPU):

  * **Process-pool preprocessing (the big win).** Audio decode + the four
    512x512 mel-spectrograms + resize per request are Python/numpy/librosa/
    torchaudio-on-CPU and **hold the GIL**, so threads give no parallelism and cap
    the whole server while the GPU starves. Running preprocessing across PROCESSES
    bypasses the GIL so it outruns the GPU -> the GPU saturates. See
    `preprocess_worker.py` (workers are CPU-only; they never load the GPU model).
  * **Dynamic batching.** UTMOSv2 inputs are FIXED shape (every clip is tiled/
    cropped to a 3 s SSL window and fixed 512x512 spectrograms), so there is no
    padding waste and no length buckets are needed. Requests are simply coalesced:
    a batch fires when it fills to MAX_BATCH or its oldest item waited MAX_WAIT_MS.
  * **Nothing CPU/GPU-bound blocks the event loop.** Preprocessing -> process pool;
    the GPU forward -> one dedicated GPU thread; result finalize (D2H wait + TTA
    average) -> thread pool. The asyncio loop only shuffles handles.
  * **Pinned H2D staging + a dedicated compute stream.** The (large) spectrogram
    tensors are copied into pinned ring buffers and pushed async; batch N's finalize
    overlaps batch N+1's compute.
  * **bf16 autocast** (matches the library's inference autocast), TF32 matmuls.
  * **Test-time augmentation.** UTMOSv2 draws random crops/mixup per pass; ?reps=N
    preprocesses N stochastic views and averages the scores (accuracy knob).

Endpoints:  POST /predict (multipart 'file'; -> {"mos": ...}) . GET /health . GET /stats
            POST /warmup . GET / (browser upload form)
"""
import asyncio
import concurrent.futures
import itertools
import multiprocessing as mp
import os
import time

import numpy as np
import torch
from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

# --------------------------------------------------------------------------- #
# config (all overridable via env)
# --------------------------------------------------------------------------- #
CONFIG        = os.environ.get("UTMOS_CONFIG", "fusion_stage3")
FOLD          = int(os.environ.get("FOLD", "0"))
MAX_BATCH     = int(os.environ.get("MAX_BATCH", "8"))
MAX_WAIT_MS   = float(os.environ.get("MAX_WAIT_MS", "10"))
NUM_FRAMES    = os.environ.get("NUM_FRAMES", "")             # "" = config default; "1" = ~4.7x fewer spec images
PIPELINE      = int(os.environ.get("PIPELINE", "1"))         # overlap finalize(N) with compute(N+1)
WARMUP        = int(os.environ.get("WARMUP", "1"))
RING          = int(os.environ.get("RING", "4"))             # pinned staging buffers in flight
PP_WORKERS    = int(os.environ.get("PP_WORKERS", "8"))       # preprocessing processes
CPU_WORKERS   = int(os.environ.get("CPU_WORKERS", "8"))      # finalize thread pool
PREDICT_DATASET = os.environ.get("PREDICT_DATASET", "sarulab")
_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
DTYPE = _DTYPES[os.environ.get("DTYPE", "bf16")]
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = bool(int(os.environ.get("CUDNN_BENCH", "1")))  # fixed shapes -> benchmark wins
torch.set_float32_matmul_precision("high")


def log(m):
    print(m, flush=True)


# --------------------------------------------------------------------------- #
# model load
# --------------------------------------------------------------------------- #
import utmosv2  # noqa: E402

log(f"[load] config={CONFIG} fold={FOLD} dtype={DTYPE} max_batch={MAX_BATCH} "
    f"wait={MAX_WAIT_MS}ms num_frames={NUM_FRAMES or 'default'} pipeline={PIPELINE}")
_m = utmosv2.create_model(pretrained=True, config=CONFIG, fold=FOLD, device=str(DEV))
CFG = _m._cfg
# Optional inference-time speed/accuracy knob (see predict()'s num_frames doc).
if NUM_FRAMES:
    sf_cfg = getattr(CFG.dataset, "spec_frames", None)
    if sf_cfg is not None:
        sf_cfg.num_frames = int(NUM_FRAMES)
MODEL = _m._model.to(DEV).eval()
for p in MODEL.parameters():
    p.requires_grad_(False)

N_FRAMES = int(getattr(CFG.dataset.spec_frames, "num_frames"))
N_SPECS  = len(CFG.dataset.specs)
SPEC_IMGS = N_FRAMES * N_SPECS                       # x2 first dim
SSL_LEN   = int(CFG.dataset.ssl.duration * CFG.sr)   # x1 length
SPEC_HW   = 512                                      # transforms resize every spec to 512x512
_AMP = DTYPE if DTYPE != torch.float32 else None

log(f"[load] loaded. ssl_len={SSL_LEN} spec_imgs={SPEC_IMGS} ({N_FRAMES} frames x {N_SPECS} specs)")


def _forward(x1, x2, d):
    """(x1[B,ssl_len], x2[B,SPEC_IMGS,3,512,512], d[B,ndom]) -> MOS [B] float32."""
    with torch.autocast("cuda", dtype=_AMP, enabled=(_AMP is not None and DEV.type == "cuda")):
        out = MODEL(x1, x2, d)
    return out.squeeze(1).float()


# --------------------------------------------------------------------------- #
# CUDA stream + pinned host ring buffers  (x2 dominates the H2D bytes)
# --------------------------------------------------------------------------- #
_use_cuda = DEV.type == "cuda"
if _use_cuda:
    S_COMP = torch.cuda.Stream()
    _PIN_X1 = [torch.zeros(MAX_BATCH, SSL_LEN, dtype=torch.float32).pin_memory() for _ in range(RING)]
    _PIN_X2 = [torch.zeros(MAX_BATCH, SPEC_IMGS, 3, SPEC_HW, SPEC_HW, dtype=torch.float32).pin_memory()
               for _ in range(RING)]
    _PIN_X1_NP = [t.numpy() for t in _PIN_X1]
    _PIN_X2_NP = [t.numpy() for t in _PIN_X2]
    _RING_EV = [torch.cuda.Event() for _ in range(RING)]
    for e in _RING_EV:
        e.record()
    _ring = itertools.cycle(range(RING))

GPU = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu")
CPU = concurrent.futures.ThreadPoolExecutor(max_workers=CPU_WORKERS, thread_name_prefix="cpu")

# Preprocessing across PROCESSES (bypass the GIL so it outruns the GPU).
try:
    import preprocess_worker

    PP = (concurrent.futures.ProcessPoolExecutor(
              max_workers=PP_WORKERS, mp_context=mp.get_context("spawn"),
              initializer=preprocess_worker.init, initargs=(CONFIG, PREDICT_DATASET))
          if PP_WORKERS > 0 else None)
except Exception as _e:
    PP = None
    print(f"[pp] ProcessPool disabled ({_e}); preprocessing on threads", flush=True)

Q: "asyncio.Queue" = None
STATS = {"requests": 0, "batches": 0, "batched_items": 0, "gpu_ms_sum": 0.0, "max_bs": 0}


# --------------------------------------------------------------------------- #
# GPU thread: build batch, launch async, return a handle (no host sync)
# --------------------------------------------------------------------------- #
def _submit(items):
    """Stack sub-items into one batch, push H2D + compute, record a done event. No host sync."""
    t0 = time.perf_counter()
    b = len(items)
    if _use_cuda:
        k = next(_ring)
        _RING_EV[k].synchronize()                       # slot k drained -> safe to reuse pinned buffers
        px1, px2 = _PIN_X1_NP[k], _PIN_X2_NP[k]
        for i, it in enumerate(items):
            px1[i] = it["x1"]
            px2[i] = it["x2"]
        d = torch.from_numpy(np.stack([it["d"] for it in items])).to(DEV, non_blocking=True)
        with torch.cuda.stream(S_COMP):
            x1 = _PIN_X1[k][:b].to(DEV, non_blocking=True)
            x2 = _PIN_X2[k][:b].to(DEV, non_blocking=True)
            out = _forward(x1, x2, d)
            out_cpu = out.to("cpu", non_blocking=True)  # small (B,) D2H
            ev = torch.cuda.Event()
            ev.record(S_COMP)
        _RING_EV[k] = ev
        return {"items": items, "out": out_cpu, "ev": ev, "t0": t0}
    else:  # CPU fallback
        x1 = torch.from_numpy(np.stack([it["x1"] for it in items]))
        x2 = torch.from_numpy(np.stack([it["x2"] for it in items]))
        d = torch.from_numpy(np.stack([it["d"] for it in items]))
        with torch.no_grad():
            out_cpu = _forward(x1, x2, d).cpu()
        return {"items": items, "out": out_cpu, "ev": None, "t0": t0}


def _finalize(h):
    """Wait for the D2H, then average the reps belonging to each request and resolve it."""
    if h["ev"] is not None:
        h["ev"].synchronize()
    gpu_ms = (time.perf_counter() - h["t0"]) * 1000.0
    out = h["out"].numpy()
    n = len(h["items"])
    # Accumulate each sub-item's score into its parent request; resolve when all reps in.
    for score, it in zip(out, h["items"]):
        req = it["req"]
        req["acc"] += float(score)
        req["got"] += 1
        if req["got"] == req["reps"]:
            req["mos"] = req["acc"] / req["reps"]
            req["gpu_ms"] = gpu_ms
            req["bs"] = n
            req["loop"].call_soon_threadsafe(req["fut"].set_result, req)
    STATS["batches"] += 1
    STATS["batched_items"] += n
    STATS["gpu_ms_sum"] += gpu_ms
    STATS["max_bs"] = max(STATS["max_bs"], n)


async def _batch_loop():
    """Coalesce queued sub-items into batches. A batch fires when it fills to MAX_BATCH
    or the oldest queued item has waited MAX_WAIT_MS. Shapes are fixed, so no bucketing."""
    loop = asyncio.get_event_loop()
    buf: list = []
    t0 = None
    prev = None
    while True:
        while len(buf) < MAX_BATCH:                      # drain everything already queued
            try:
                buf.append(Q.get_nowait())
                if t0 is None:
                    t0 = time.monotonic()
            except asyncio.QueueEmpty:
                break
        fire = len(buf) >= MAX_BATCH or (buf and (time.monotonic() - t0) * 1000 >= MAX_WAIT_MS)
        if fire:
            items = buf[:MAX_BATCH]
            del buf[:MAX_BATCH]
            t0 = time.monotonic() if buf else None
            handle = await loop.run_in_executor(GPU, _submit, items)
            if PIPELINE:
                if prev is not None:
                    loop.run_in_executor(CPU, _finalize, prev)
                prev = handle
            else:
                await loop.run_in_executor(CPU, _finalize, handle)
            continue
        if prev is not None:                             # nothing ready: flush the pending handle
            loop.run_in_executor(CPU, _finalize, prev)
            prev = None
        if buf:                                          # wait only until the oldest item's deadline
            timeout = max(0.001, MAX_WAIT_MS / 1000 - (time.monotonic() - t0))
            try:
                buf.append(await asyncio.wait_for(Q.get(), timeout))
            except asyncio.TimeoutError:
                pass
        else:
            buf.append(await Q.get())
            t0 = time.monotonic()


# --------------------------------------------------------------------------- #
def _warmup(batch_sizes=(1, MAX_BATCH)):
    if not _use_cuda:
        return
    with torch.no_grad():
        for b in sorted(set(batch_sizes)):
            x1 = torch.zeros(b, SSL_LEN, device=DEV)
            x2 = torch.zeros(b, SPEC_IMGS, 3, SPEC_HW, SPEC_HW, device=DEV)
            d = torch.zeros(b, MODEL_NDOM, device=DEV)
            with torch.cuda.stream(S_COMP):
                _forward(x1, x2, d)
    torch.cuda.synchronize()
    log(f"[warmup] primed batch sizes {sorted(set(batch_sizes))}")


# Probe the domain-embedding width once (needed for the warmup dummy tensor).
try:
    from utmosv2.dataset._utils import get_dataset_map

    MODEL_NDOM = len(get_dataset_map(CFG))
except Exception:
    MODEL_NDOM = None

# --------------------------------------------------------------------------- #
app = FastAPI(title="utmosv2-serve")


@app.on_event("startup")
async def _startup():
    global Q
    Q = asyncio.Queue()
    loop = asyncio.get_event_loop()
    if WARMUP and MODEL_NDOM is not None:
        await loop.run_in_executor(GPU, _warmup)
    if PP is not None:                                   # spin up + prime the preprocessing procs
        try:
            import io as _io

            import soundfile as sf

            sil = _io.BytesIO()
            # an audible tone (not silence: silence-removal would empty pure quiet)
            _t = np.arange(CFG.sr, dtype="float32") / CFG.sr
            sf.write(sil, (0.3 * np.sin(2 * np.pi * 220 * _t)).astype("float32"), CFG.sr, format="WAV")
            blob = sil.getvalue()
            await asyncio.gather(*[
                loop.run_in_executor(PP, preprocess_worker.preprocess, blob, PREDICT_DATASET, 1)
                for _ in range(PP_WORKERS)
            ])
            log(f"[startup] preprocessing ProcessPool ready ({PP_WORKERS} workers)")
        except Exception as e:
            log(f"[startup] ProcessPool warm failed: {e}")
    asyncio.create_task(_batch_loop())
    log("[startup] ready")


@app.get("/health")
async def health():
    return {"ok": True, "config": CONFIG, "fold": FOLD, "dtype": str(DTYPE),
            "device": str(DEV), "max_batch": MAX_BATCH, "max_wait_ms": MAX_WAIT_MS,
            "num_frames": N_FRAMES, "spec_imgs": SPEC_IMGS,
            "pp_workers": PP_WORKERS if PP is not None else 0}


@app.get("/stats")
async def stats():
    b = max(1, STATS["batches"])
    return {**STATS, "avg_batch": STATS["batched_items"] / b, "avg_gpu_ms": STATS["gpu_ms_sum"] / b}


@app.post("/warmup")
async def warmup():
    await asyncio.get_event_loop().run_in_executor(GPU, _warmup)
    return await health()


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    reps: int = Query(1, ge=1, le=16, description="TTA passes to average (accuracy knob)"),
    dataset: str = Query(None, description="data-domain name (defaults to server PREDICT_DATASET)"),
):
    raw = await file.read()
    loop = asyncio.get_event_loop()
    ds = dataset or PREDICT_DATASET
    exec_ = PP if PP is not None else CPU
    x1, x2, d = await loop.run_in_executor(exec_, preprocess_worker.preprocess, raw, ds, reps)

    req = {"reps": x1.shape[0], "got": 0, "acc": 0.0,
           "fut": loop.create_future(), "loop": loop}
    STATS["requests"] += 1
    t0 = time.perf_counter()
    for i in range(x1.shape[0]):                          # enqueue each TTA view as a sub-item
        await Q.put({"x1": x1[i], "x2": x2[i], "d": d[i], "req": req})
    await req["fut"]
    total_ms = (time.perf_counter() - t0) * 1000
    return JSONResponse({
        "mos": req["mos"],
        "reps": req["reps"],
        "gpu_ms": round(req["gpu_ms"], 2),
        "total_ms": round(total_ms, 2),
        "batch_size": req["bs"],
    })


PLAYER_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>UTMOSv2 — MOS prediction</title>
<style>
 :root{color-scheme:light dark;--fg:#111;--muted:#666;--bg:#fafafa;--card:#fff;--line:#e3e3e3;--accent:#2563eb}
 @media (prefers-color-scheme:dark){:root{--fg:#e8e8e8;--muted:#9aa0a6;--bg:#0f1114;--card:#181b1f;--line:#2a2e34;--accent:#5b8cff}}
 *{box-sizing:border-box}body{font:15px/1.5 system-ui,sans-serif;margin:0;background:var(--bg);color:var(--fg)}
 .wrap{max-width:640px;margin:0 auto;padding:32px 20px}
 h1{font-size:22px;margin:0 0 4px}.sub{color:var(--muted);margin:0 0 24px}
 .card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:20px;margin:16px 0}
 input[type=file]{width:100%;padding:12px;border:1px dashed var(--line);border-radius:8px;background:transparent;color:var(--fg)}
 button{margin-top:14px;padding:10px 18px;border:0;border-radius:8px;background:var(--accent);color:#fff;font-weight:600;cursor:pointer}
 button:disabled{opacity:.5}audio{width:100%;margin-top:8px}
 .score{font-size:44px;font-weight:700;margin:10px 0}.chip{font-size:12px;background:var(--bg);border:1px solid var(--line);border-radius:999px;padding:4px 10px;color:var(--muted);margin-right:6px}
 .err{color:#dc2626;margin-top:12px}
</style></head><body><div class="wrap">
<h1>UTMOSv2 — MOS prediction</h1>
<p class="sub">Upload a speech clip; get its predicted naturalness MOS (~1–5).</p>
<div class="card">
 <input id="f" type="file" accept="audio/*,.wav,.mp3,.flac,.ogg">
 <audio id="a" controls style="display:none"></audio>
 <div><button id="go" disabled>Predict MOS</button></div>
 <div id="score" class="score"></div>
 <div id="meta"></div>
 <div id="err" class="err"></div>
</div></div><script>
const $=id=>document.getElementById(id);
$('f').onchange=e=>{const f=e.target.files[0];$('go').disabled=!f;if(f){const a=$('a');a.src=URL.createObjectURL(f);a.style.display='block';$('score').textContent='';$('meta').innerHTML='';$('err').textContent='';}};
$('go').onclick=async()=>{const f=$('f').files[0];if(!f)return;$('go').disabled=true;$('go').textContent='Predicting…';$('err').textContent='';
 try{const fd=new FormData();fd.append('file',f,f.name);const t0=performance.now();
  const r=await fetch('predict',{method:'POST',body:fd});if(!r.ok)throw new Error('HTTP '+r.status+' '+(await r.text()).slice(0,200));
  const j=await r.json();$('score').textContent=j.mos.toFixed(3);
  $('meta').innerHTML=[['round-trip',Math.round(performance.now()-t0)+' ms'],['gpu',j.gpu_ms+' ms'],['batch',j.batch_size],['reps',j.reps]].map(([k,v])=>`<span class="chip">${k}: <b>${v}</b></span>`).join('');
 }catch(e){$('err').textContent='⚠ '+e.message;}finally{$('go').disabled=false;$('go').textContent='Predict MOS';}};
</script></body></html>"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def player():
    return PLAYER_HTML
