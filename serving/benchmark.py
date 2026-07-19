#!/usr/bin/env python3
"""Concurrency + throughput benchmark for the UTMOSv2 serving API.

Fires TOTAL requests over a folder of clips, capped at C in flight, for each C in
CONC. Reports, per level:
  * throughput (req/s)
  * RTF = audio-seconds scored / wall-second
  * client latency p50 / p95 / p99 (ms)
  * avg / max server-side batch (from the JSON batch_size -> dynamic batching)

  URL=http://127.0.0.1:8000 \
  CLIPS='../Elise_audio/*.wav' \
  CONC=1,4,8,16,32 TOTAL=200 REPS=1 python benchmark.py
"""
import asyncio
import glob
import os
import statistics
import time

import httpx
import soundfile as sf

URL   = os.environ.get("URL", "http://127.0.0.1:8000")
CONC  = [int(x) for x in os.environ.get("CONC", "1,4,8,16,32").split(",")]
TOTAL = int(os.environ.get("TOTAL", "200"))
REPS  = int(os.environ.get("REPS", "1"))
GLOB  = os.environ.get("CLIPS", "../Elise_audio/*.wav")

FILES = sorted(f for pat in GLOB.split(":") for f in glob.glob(pat))
assert FILES, f"no clips under {GLOB}"
PAYLOADS = []
for f in FILES:
    try:
        info = sf.info(f)
        dur = info.frames / info.samplerate
    except Exception:
        dur = 0.0
    PAYLOADS.append((os.path.basename(f), open(f, "rb").read(), dur))
AUDIO_TOTAL = sum(d for _, _, d in PAYLOADS)


async def _one(client, sem, payload, out):
    name, blob, dur = payload
    async with sem:
        t0 = time.perf_counter()
        try:
            r = await client.post(
                f"{URL}/predict?reps={REPS}",
                files={"file": (name, blob, "application/octet-stream")},
                timeout=300,
            )
            r.raise_for_status()
            j = r.json()
            out.append((time.perf_counter() - t0, dur, j.get("batch_size", 0)))
        except Exception as e:
            out.append((time.perf_counter() - t0, dur, -1))
            print(f"  ! {name}: {e}")


async def _run(conc):
    sem = asyncio.Semaphore(conc)
    out: list = []
    payloads = [PAYLOADS[i % len(PAYLOADS)] for i in range(TOTAL)]
    audio_s = sum(p[2] for p in payloads)
    async with httpx.AsyncClient() as client:
        t0 = time.perf_counter()
        await asyncio.gather(*[_one(client, sem, p, out) for p in payloads])
        wall = time.perf_counter() - t0
    lat = sorted(o[0] * 1000 for o in out if o[2] >= 0)
    bss = [o[2] for o in out if o[2] > 0]
    ok = len(lat)
    if not lat:
        print(f"C={conc:<3} all failed")
        return

    def pct(p):
        return lat[min(len(lat) - 1, int(len(lat) * p))]

    print(f"C={conc:<3} ok={ok}/{TOTAL}  {ok / wall:6.1f} req/s  "
          f"RTF={audio_s / wall:6.1f}x  "
          f"p50={statistics.median(lat):6.1f}  p95={pct(0.95):6.1f}  p99={pct(0.99):6.1f} ms  "
          f"batch avg={statistics.mean(bss):.1f} max={max(bss)}")


async def main():
    print(f"{len(FILES)} clips, {AUDIO_TOTAL:.1f} s audio; TOTAL={TOTAL} REPS={REPS} -> {URL}")
    for c in CONC:
        await _run(c)


if __name__ == "__main__":
    asyncio.run(main())
