#!/usr/bin/env python3
"""CPU-vs-GPU feature extraction benchmark for UTMOSv2 serving.

Compares the CPU worker feature stack (SSLLMultiSpecExtDataset, what
FEATURE_DEVICE=cpu runs per process-pool worker) against the batched GPU
featurizer (gpu_features.GpuFeaturizer, what FEATURE_DEVICE=cuda runs):

  1. SPEED   — per-clip featurization latency (decode excluded, timed separately),
               single CPU worker (torch threads=1, matching serving) vs one GPU call.
  2. ACCURACY— np.random is seeded identically for both paths, so crops + mixup
               lambdas are IDENTICAL and any difference is pure DSP numerics:
               feature tensor deltas, plus MOS through the real model (bf16, like
               serving) for both feature sets. A CPU-vs-CPU different-seed run
               gives the natural TTA noise floor for context.

Run on the GPU box:
  python benchmark_features.py --clips '/root/emgs_stress/*.mp3' --n 100
"""
from __future__ import annotations

import argparse
import csv
import glob
import importlib
import statistics
import time
from types import SimpleNamespace

import numpy as np
import torch

import preprocess_worker  # decode_clean only; init() is never called here
from gpu_features import GpuFeaturizer


def build_cfg(config: str):
    from utmosv2._settings import configure_execution

    mod = importlib.import_module(f"utmosv2.config.{config}")
    cfg = SimpleNamespace(**{k: v for k, v in mod.__dict__.items() if not k.startswith("__")})
    configure_execution(cfg)
    # clips are pre-cleaned with decode_clean, so the dataset must not re-trim
    cfg.dataset.remove_silent_section = False
    return cfg


def build_cpu_dataset(cfg, dataset_name: str):
    from utmosv2.dataset._schema import InMemoryData
    from utmosv2.utils import get_dataset

    dummy = InMemoryData(data=np.zeros(16000, dtype=np.float32), dataset_name=dataset_name)
    return get_dataset(cfg, dummy, cfg.phase)


def cpu_featurize(ds, audio: np.ndarray, dataset_name: str):
    """One rep through the exact CPU worker path -> (x1, x2, d) torch CPU tensors."""
    from utmosv2.dataset._schema import InMemoryData

    data = InMemoryData(data=audio, dataset_name=dataset_name)
    ds.data = data
    ds.ssl.data = data
    ds.multi_spec.data = data
    x1, x2, d, _ = ds[0]
    return x1, x2, d


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="/root/emgs_stress/*.mp3")
    ap.add_argument("--n", type=int, default=100, help="clips for the accuracy comparison")
    ap.add_argument("--speed-n", type=int, default=20, help="clips for the speed timing")
    ap.add_argument("--iters", type=int, default=3, help="timing iterations per clip")
    ap.add_argument("--reps", type=int, default=5, help="TTA reps for the reps-batched timing")
    ap.add_argument("--config", default="fusion_stage3")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--dataset", default="sarulab")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="features_cpu_vs_gpu.csv")
    args = ap.parse_args()

    # one CPU thread = one serving worker's conditions (workers set OMP_NUM_THREADS=1)
    torch.set_num_threads(1)
    dev = torch.device(args.device)
    amp = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[args.dtype]

    files = sorted(glob.glob(args.clips))
    assert files, f"no clips match {args.clips}"
    files = files[: max(args.n, args.speed_n)]
    print(f"[setup] {len(files)} clips, config={args.config} fold={args.fold} dtype={args.dtype}")

    print("[setup] decoding clips (decode_clean: decode+resample+silence-trim) ...")
    audios, dec_ms = [], []
    for f in files:
        raw = open(f, "rb").read()
        t0 = time.perf_counter()
        audios.append(preprocess_worker.decode_clean(raw))
        dec_ms.append((time.perf_counter() - t0) * 1000)

    cfg = build_cfg(args.config)
    ds = build_cpu_dataset(cfg, args.dataset)
    feat = GpuFeaturizer(cfg, device=dev)

    import utmosv2

    print("[setup] loading model ...")
    m = utmosv2.create_model(pretrained=True, config=args.config, fold=args.fold, device=str(dev))
    model = m._model.to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    def forward_mos(x1, x2, d):
        with torch.no_grad(), torch.autocast(dev.type, dtype=amp, enabled=amp is not None):
            out = model(x1.to(dev), x2.to(dev), d.to(dev))
        return out.squeeze(1).float().cpu().numpy()

    # ---------------- speed ----------------
    print("\n================ SPEED (featurize only, decode excluded) ================")
    sn = min(args.speed_n, len(audios))
    for a in audios[:2]:  # warmup both paths (cuFFT plans, resize kernels, caches)
        cpu_featurize(ds, a, args.dataset)
        feat.featurize(a, args.dataset, 1)
        feat.featurize(a, args.dataset, args.reps)
    if dev.type == "cuda":
        torch.cuda.synchronize()

    cpu1, gpu1, cpuR, gpuR = [], [], [], []
    for a in audios[:sn]:
        for _ in range(args.iters):
            t0 = time.perf_counter()
            cpu_featurize(ds, a, args.dataset)
            cpu1.append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            for _ in range(args.reps):
                cpu_featurize(ds, a, args.dataset)
            cpuR.append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            feat.featurize(a, args.dataset, 1)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            gpu1.append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            feat.featurize(a, args.dataset, args.reps)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            gpuR.append((time.perf_counter() - t0) * 1000)

    def row(name, xs):
        print(f"  {name:<28} mean={statistics.mean(xs):8.2f}  p50={statistics.median(xs):8.2f}  "
              f"p95={pct(xs, 0.95):8.2f} ms")

    row("decode+trim (per clip)", dec_ms)
    row("CPU featurize reps=1", cpu1)
    row(f"CPU featurize reps={args.reps}", cpuR)
    row("GPU featurize reps=1", gpu1)
    row(f"GPU featurize reps={args.reps}", gpuR)
    print(f"  -> GPU speedup: reps=1 {statistics.median(cpu1) / statistics.median(gpu1):.1f}x, "
          f"reps={args.reps} {statistics.median(cpuR) / statistics.median(gpuR):.1f}x  "
          f"(single CPU worker; serving runs PP_WORKERS of them)")

    # ---------------- accuracy ----------------
    print("\n================ ACCURACY (identical seeds -> identical crops) ================")
    an = min(args.n, len(audios))
    x2_max, x2_mean = 0.0, []
    x1_identical = True
    mos_cpu, mos_gpu = [], []
    B = 8
    buf_c, buf_g = [], []

    def flush():
        if not buf_c:
            return
        x1c = torch.stack([b[0] for b in buf_c]); x2c = torch.stack([b[1] for b in buf_c])
        dc = torch.stack([b[2] for b in buf_c])
        mos_cpu.extend(forward_mos(x1c, x2c, dc))
        x1g = torch.stack([b[0] for b in buf_g]); x2g = torch.stack([b[1] for b in buf_g])
        dg = torch.stack([b[2] for b in buf_g])
        mos_gpu.extend(forward_mos(x1g, x2g, dg))
        buf_c.clear(); buf_g.clear()

    for i in range(an):
        seed = 1234 + i
        np.random.seed(seed)
        x1c, x2c, dc = cpu_featurize(ds, audios[i], args.dataset)
        np.random.seed(seed)
        x1g, x2g, dg = feat.featurize(audios[i], args.dataset, 1)
        x1g, x2g, dg = x1g[0].cpu(), x2g[0].cpu(), dg[0].cpu()

        x1_identical &= torch.equal(x1c, x1g)
        dx2 = (x2g - x2c).abs()
        x2_max = max(x2_max, dx2.max().item())
        x2_mean.append(dx2.mean().item())

        buf_c.append((x1c, x2c, dc)); buf_g.append((x1g, x2g, dg))
        if len(buf_c) == B:
            flush()
    flush()

    mos_cpu = np.array(mos_cpu); mos_gpu = np.array(mos_gpu)
    diff = mos_gpu - mos_cpu
    print(f"  clips                : {an}")
    print(f"  x1 (SSL wave)        : {'bit-identical' if x1_identical else 'DIFFERS (bug!)'}")
    print(f"  x2 (spectrograms)    : max|d|={x2_max:.3e}  mean|d|={np.mean(x2_mean):.3e}  "
          f"(range of x2 is ~[0, 1])")
    print(f"  MOS  ({args.dtype} forward) : MAE={np.mean(np.abs(diff)):.6f}  max|d|={np.max(np.abs(diff)):.6f}  "
          f"bias={np.mean(diff):+.6f}  Pearson r={np.corrcoef(mos_cpu, mos_gpu)[0, 1]:.6f}")

    # natural TTA noise floor: same CPU path, two different seeds
    nf = min(30, an)
    a_mos, b_mos = [], []
    for i in range(nf):
        np.random.seed(50_000 + i)
        x1a, x2a, da = cpu_featurize(ds, audios[i], args.dataset)
        np.random.seed(90_000 + i)
        x1b, x2b, db = cpu_featurize(ds, audios[i], args.dataset)
        a_mos.extend(forward_mos(x1a[None], x2a[None], da[None]))
        b_mos.extend(forward_mos(x1b[None], x2b[None], db[None]))
    nfd = np.abs(np.array(a_mos) - np.array(b_mos))
    print(f"  noise floor (CPU vs CPU, different seeds, {nf} clips, reps=1): "
          f"MAE={nfd.mean():.4f}  max={nfd.max():.4f}")

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "mos_cpu_features", "mos_gpu_features"])
        for i in range(an):
            w.writerow([files[i], f"{mos_cpu[i]:.6f}", f"{mos_gpu[i]:.6f}"])
    print(f"\n[done] per-clip MOS pairs -> {args.out}")


if __name__ == "__main__":
    main()
