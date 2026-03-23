"""
Stress test for UTMOSv2 inference using audio files in Elise_audio/.

Tests:
  1. Single WAV / MP3 file inference
  2. WAV vs MP3 score agreement (matched pairs)
  3. Score consistency (same file, multiple runs)
  4. Batch throughput: WAV directory
  5. Batch throughput: MP3 directory
  6. Batch throughput: mixed WAV+MP3 directory

Usage:
    python3 stress_test.py
    python3 stress_test.py --audio_dir Elise_audio --num_files 50 --compile
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import librosa
import numpy as np
import torch

import utmosv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", type=str, default="Elise_audio")
    parser.add_argument("--num_files", type=int, default=None,
                        help="Max files per format to use (None = all)")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_repetitions", type=int, default=1)
    parser.add_argument("--compile", action="store_true",
                        help="Enable torch.compile on the model (requires PyTorch >= 2.0)")
    return parser.parse_args()


def audio_duration(file: Path) -> float:
    try:
        import soundfile as sf
        info = sf.info(str(file))
        return info.duration
    except Exception:
        try:
            return librosa.get_duration(filename=str(file))
        except Exception:
            return 0.0


def total_duration(files: list[Path]) -> float:
    return sum(audio_duration(f) for f in files)


def sep(title: str = "") -> None:
    w = 62
    if title:
        pad = (w - len(title) - 2) // 2
        print("=" * pad + f" {title} " + "=" * (w - pad - len(title) - 2))
    else:
        print("=" * w)


def predict_single(model, f, args, device):
    return model.predict(
        input_path=f,
        device=device,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        num_repetitions=args.num_repetitions,
        verbose=False,
    )


def bench_files(
    label: str,
    model,
    files: list[Path],
    args,
    device: torch.device,
) -> dict:
    sep(label)
    print(f"  Files : {len(files)}")
    dur = total_duration(files)
    print(f"  Audio : {dur:.1f}s  ({dur / 60:.1f} min)")

    t0 = time.perf_counter()
    scores = [predict_single(model, f, args, device) for f in files]
    elapsed = time.perf_counter() - t0

    scores = np.array(scores)
    rtf = dur / elapsed if elapsed > 0 else float("inf")
    tput = len(files) / elapsed

    print(f"  Elapsed    : {elapsed:.2f}s")
    print(f"  Throughput : {tput:.2f} files/sec  |  RTF: {rtf:.1f}x")
    print(f"  Score      : mean={scores.mean():.4f}  std={scores.std():.4f}"
          f"  [{scores.min():.4f}, {scores.max():.4f}]")
    print()
    return dict(label=label, n=len(files), dur=dur, elapsed=elapsed, tput=tput, rtf=rtf)


def bench_dir(
    label: str,
    model,
    input_dir: Path,
    args,
    device: torch.device,
) -> dict:
    sep(label)
    t0 = time.perf_counter()
    results = model.predict(
        input_dir=input_dir,
        device=device,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        num_repetitions=args.num_repetitions,
        verbose=True,
    )
    elapsed = time.perf_counter() - t0

    scores = np.array([r["predicted_mos"] for r in results])
    files = [Path(r["file_path"]) for r in results]
    dur = total_duration(files)
    rtf = dur / elapsed if elapsed > 0 else float("inf")
    tput = len(results) / elapsed

    print(f"  Files      : {len(results)}")
    print(f"  Audio      : {dur:.1f}s  ({dur / 60:.1f} min)")
    print(f"  Elapsed    : {elapsed:.2f}s")
    print(f"  Throughput : {tput:.2f} files/sec  |  RTF: {rtf:.1f}x")
    print(f"  Score      : mean={scores.mean():.4f}  std={scores.std():.4f}"
          f"  [{scores.min():.4f}, {scores.max():.4f}]")
    print()
    return dict(label=label, n=len(results), dur=dur, elapsed=elapsed, tput=tput, rtf=rtf)


def main() -> None:
    args = parse_args()
    audio_dir = Path(args.audio_dir)
    assert audio_dir.exists(), f"Not found: {audio_dir}"

    wav_files = sorted(audio_dir.glob("*.wav"))
    mp3_files = sorted(audio_dir.glob("*.mp3"))
    if args.num_files is not None:
        wav_files = wav_files[: args.num_files]
        mp3_files = mp3_files[: args.num_files]

    assert wav_files, f"No WAV files in {audio_dir}"
    assert mp3_files, f"No MP3 files in {audio_dir}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice     : {device}")
    print(f"WAV files  : {len(wav_files)}")
    print(f"MP3 files  : {len(mp3_files)}")
    print(f"compile    : {args.compile}\n")

    print("Loading model...")
    model = utmosv2.create_model(compile=args.compile)
    print()

    summary = []

    # ── 1. Single file (warm-up + single-file overhead) ────────────────────
    for label, f in [("Single WAV", wav_files[0]), ("Single MP3", mp3_files[0])]:
        r = bench_files(label, model, [f], args, device)
        summary.append(r)

    # ── 2. Score consistency (same file, 5 runs) ───────────────────────────
    sep("Score consistency — same WAV x5")
    reps = [predict_single(model, wav_files[0], args, device) for _ in range(5)]
    print(f"  File  : {wav_files[0].name}")
    print(f"  Scores: {[f'{s:.4f}' for s in reps]}")
    print(f"  Std   : {np.std(reps):.4f}  (expected ~0.1–0.2 with num_repetitions=1)\n")

    # ── 3. WAV vs MP3 agreement ────────────────────────────────────────────
    sep("WAV vs MP3 agreement (10 matched pairs)")
    wav_map = {f.stem: f for f in wav_files[:20]}
    mp3_map = {f.stem: f for f in mp3_files[:20]}
    pairs = sorted(set(wav_map) & set(mp3_map))[:10]
    diffs = []
    for stem in pairs:
        sw = predict_single(model, wav_map[stem], args, device)
        sm = predict_single(model, mp3_map[stem], args, device)
        diffs.append(abs(sw - sm))
        print(f"  {stem[:48]:48s}  WAV={sw:.4f}  MP3={sm:.4f}  Δ={abs(sw - sm):.4f}")
    print(f"\n  Mean |WAV − MP3| : {np.mean(diffs):.4f}  "
          f"(non-zero due to random TTA crop)\n")

    # ── 4. Batch throughput: per-file loop ─────────────────────────────────
    r = bench_files(f"Batch WAV {len(wav_files)} files (per-file)",
                    model, wav_files, args, device)
    summary.append(r)

    r = bench_files(f"Batch MP3 {len(mp3_files)} files (per-file)",
                    model, mp3_files, args, device)
    summary.append(r)

    # ── 5. Throughput with temp dirs (wav-only / mp3-only) ─────────────────
    # Create symlink temp dirs so we can use the faster input_dir API.
    import tempfile, os
    for fmt_label, files in [("WAV dir", wav_files), ("MP3 dir", mp3_files)]:
        with tempfile.TemporaryDirectory() as tmp:
            for f in files:
                os.symlink(f.resolve(), Path(tmp) / f.name)
            r = bench_dir(f"Batch {fmt_label} (input_dir API)",
                          model, Path(tmp), args, device)
            summary.append(r)

    # ── Summary ────────────────────────────────────────────────────────────
    sep("SUMMARY")
    print(f"  {'Test':<42} {'Files':>6} {'Elapsed':>9} {'Files/s':>8} {'RTF':>7}")
    print("  " + "-" * 76)
    for r in summary:
        print(f"  {r['label']:<42} {r['n']:>6} "
              f"{r['elapsed']:>8.2f}s {r['tput']:>8.2f} {r['rtf']:>6.1f}x")
    print()


if __name__ == "__main__":
    main()
