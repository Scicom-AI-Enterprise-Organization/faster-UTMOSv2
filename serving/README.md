# UTMOSv2 serving — high-throughput MOS prediction

A FastAPI inference server for [UTMOSv2](../README.md) that turns the
single-file `model.predict(...)` path into a **high-throughput HTTP service**, using
the same throughput stack as a production TTS/codec serving pipeline:

- **Process-pool preprocessing** — the real bottleneck, and the big win.
- **Dynamic batching** — coalesce concurrent requests into GPU batches.
- **A single dedicated GPU thread** — the asyncio event loop never blocks.

```
bytes ─► [process pool] decode+resample+4× mel-spec+resize ─► (x1, x2, d)
      ─► [async queue]  dynamic batching (fill MAX_BATCH or wait MAX_WAIT_MS)
      ─► [GPU thread]   fusion model (SSL + EfficientNetV2) ─► MOS
      ─► [thread pool]  D2H + TTA average ─► JSON {"mos": ...}
```

## Why this shape

UTMOSv2's per-request CPU work is **heavy and GIL-bound**: every clip is decoded,
silence-trimmed, and turned into **four 512×512 mel-spectrograms per frame** (8
spectrogram images by default) plus a 3 s SSL waveform. That is Python / numpy /
librosa / torchaudio-on-CPU, all holding the GIL — so running it in *threads* gives
no real parallelism and **starves the GPU** (measured ~45 % GPU utilisation, batches
never filling past ~3). Running preprocessing across **processes** bypasses the GIL,
lets it outrun the GPU, and the GPU saturates → batches fill → **~3× throughput**.

Unlike variable-length codec models, **UTMOSv2 inputs are fixed shape** (every clip is
tiled/cropped to a 3 s SSL window and fixed 512×512 spectrograms), so there is no
padding waste and **no length bucketing is needed** — batching is a simple count-based
coalesce. `cudnn.benchmark` is left ON because the shapes never change.

## Endpoints

| Method | Path       | Notes |
|--------|------------|-------|
| POST   | `/predict` | multipart `file`; `?reps=N` averages N stochastic TTA passes; `?dataset=` sets the data-domain. Returns `{"mos", "reps", "gpu_ms", "total_ms", "batch_size"}`. |
| GET    | `/`        | browser upload form (drop a clip, see its MOS). |
| GET    | `/health`  | config / device / batching knobs. |
| GET    | `/stats`   | running batch/latency counters. |
| POST   | `/warmup`  | re-prime the GPU shapes. |

## Run

```bash
pip install -e ..                 # the utmosv2 package (torch, librosa, timm, …)
pip install -r requirements.txt   # fastapi, uvicorn, soundfile, soxr, httpx

# weights auto-download from HuggingFace on first start
MAX_BATCH=16 PP_WORKERS=24 bash run_serve.sh
# -> http://127.0.0.1:8000  (open / in a browser, or POST /predict)
```

Single request:

```bash
curl -X POST http://127.0.0.1:8000/predict -F file=@clip.wav
# {"mos":3.87,"reps":1,"gpu_ms":105.5,"total_ms":136.0,"batch_size":1}
```

### Tuning knobs (env)

| Var | Default | Meaning |
|-----|---------|---------|
| `FEATURE_DEVICE` | cuda | feature stack on the GPU (see below); workers only decode+silence-trim. `cpu` = process-pool feature path (also the automatic fallback without CUDA). |
| `PP_WORKERS`  | 8    | preprocessing **processes**. The main throughput dial — raise it until the GPU saturates (needs CPU cores). |
| `MAX_BATCH`   | 8    | max GPU batch. 16 is a good sweet spot; larger mainly adds latency. |
| `MAX_WAIT_MS` | 10   | how long the oldest queued item waits for a batch to fill. 20–30 ms helps batches fill under bursty load. |
| `DTYPE`       | bf16 | `bf16` \| `fp16` \| `fp32` autocast for the forward. |
| `NUM_FRAMES`  | (config) | set `1` to halve the spectrogram images per clip (~fewer FLOPs, minor accuracy trade-off). |
| `PP_WORKERS=0`|      | disable the process pool (preprocess on threads) for debugging. |

## Benchmark

```bash
CLIPS='../audio/*.mp3' CONC=16,32,64,96 TOTAL=400 python benchmark.py
```

Measured on **1× NVIDIA H20 (shared box)**, two clips (10.8 s + 19.2 s) cycled,
`RTF = audio-seconds scored / wall-second`:

| Config | Concurrency | req/s | **RTF** | batch avg (max) | GPU |
|--------|-------------|-------|---------|-----------------|-----|
| `PP=6, batch=8`  (under-provisioned) | 32 | 14.2 | **213×** | 2.6 (5)  | ~45 % (starved) |
| `PP=24, batch=16` | 64 | 40.0 | **601×** | 14.4 (16) | saturated |
| `PP=32, batch=24` | 96 | 43.4 | **652×** | 18.1 (21) | saturated |

Provisioning the preprocessing pool to outrun the GPU (and letting batches fill) takes
throughput from **213× → ~650× RTF (~3×)** — the server flips from preprocessing-bound
to GPU-bound, exactly as intended. (Absolute numbers vary; the H20 box was shared.)

## GPU feature extraction (`FEATURE_DEVICE=cuda`, the default)

With `FEATURE_DEVICE=cpu` (or no CUDA available) the whole feature stack runs on CPU
in the process pool. With `FEATURE_DEVICE=cuda` (default), workers only **decode +
silence-trim**, and
`gpu_features.py` runs the DSP (4× mel-STFT + inner mixup + resize) as a few **batched
CUDA kernels** on the model's GPU, on its own stream: only ~1 MB/rep of raw waveform
crops crosses PCIe instead of ~25 MB/rep of finished 512×512 spectrogram images, and
the pinned staging rings aren't needed at all.

### Speed (1× H20, 164-core shared box, 183 real 5–30 s call-centre mp3 clips)

Featurize only, per clip (`benchmark_features.py`; decode ≈ 9 ms is excluded and
identical for both):

| Path | reps=1 (p50) | reps=5 (p50) |
|------|-------------|--------------|
| CPU (one worker, 1 thread — what each pool process does) | 357 ms | 1789 ms |
| GPU (batched kernels) | **4.0 ms** | **11.5 ms** |
| speedup | **89×** | **155×** |

End-to-end HTTP (`benchmark.py`, `MAX_BATCH=16 MAX_WAIT_MS=20 PP_WORKERS=24`, same
clips, reps=1; two measurement rounds because the box's background load shifted):

| Concurrency | `FEATURE_DEVICE=cpu` | `FEATURE_DEVICE=cuda` | gain |
|-------------|----------------------|------------------------|------|
| 1 (latency) | p50 527 ms | p50 **120 ms** | 4.4× |
| 8   | 10.4 req/s (RTF 168×), p50 646 ms | 22.3 req/s (RTF 361×), p50 226 ms | 2.1× |
| 32  | 21.9 req/s (RTF 356×) | 42.1 req/s (RTF 681×) | 1.9× |
| 64 (round 1) | 41.8 req/s (**RTF 677×**) | 66.6 req/s (**RTF 1078×**) | 1.6× |
| 64 (round 2, busier box) | 24.3 req/s (RTF 393×) | 41.9 req/s (RTF 678×) | 1.7× |

The CPU path is preprocessing-bound (24 workers × ~2.8 clips/s); the GPU path removes
that bottleneck entirely — one GPU featurizes ~250 clips/s — and becomes model-bound,
so it also frees ~24 cores of CPU.

### Accuracy (same box, 100 clips)

`gpu_features.py` consumes `np.random` in exactly the dataset's order, so seeding both
paths identically yields **identical crops/mixup** and isolates pure DSP numerics
(TF32 is forced off for the mel matmul; cuFFT vs CPU FFT and GPU resize remain):

| Metric | CPU vs GPU features (same seed) |
|--------|--------------------------------|
| SSL waveform `x1` | bit-identical |
| Spectrograms `x2` (range ≈ [0, 1]) | max Δ 3.9e-5, mean Δ 8e-8 |
| MOS through the model (bf16) | MAE **0.0009**, max 0.0078, Pearson r **0.99999** |
| — context: CPU-vs-CPU with *different* seeds (natural TTA noise, reps=1) | MAE 0.13, max 0.77 |

The GPU features shift MOS ~150× less than the model's own random-crop noise floor,
i.e. the accuracy difference is negligible.

## Files

- `server.py` — the FastAPI app: dynamic-batch loop, GPU thread, pinned H2D staging.
- `preprocess_worker.py` — CPU-only process-pool worker; reuses the exact UTMOSv2
  `SSLLMultiSpecExtDataset` feature stack, so served scores match the library path.
  (It decodes with soundfile/librosa, so it also works where `torchaudio.load` needs
  `torchcodec`.)
- `gpu_features.py` — batched GPU featurizer (`FEATURE_DEVICE=cuda`), numerically
  matched to the CPU worker path.
- `run_serve.sh` — launcher with sane env defaults.
- `benchmark.py` — async concurrency / RTF benchmark client.
- `benchmark_features.py` — CPU-vs-GPU featurization benchmark (speed + seeded
  numerical/MOS accuracy comparison).
