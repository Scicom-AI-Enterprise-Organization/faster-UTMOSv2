"""
Accuracy benchmark: compare this fork's predictions against the upstream
sarulab-speech/UTMOSv2.

Workflow
--------
Step 1 — generate predictions from THIS fork (run as normal):
    python benchmark_accuracy.py --audio_dir Elise_audio --num_files 100 \
        --out ours.csv

Step 2 — generate predictions from the UPSTREAM original:
    pip install git+https://github.com/sarulab-speech/UTMOSv2 --target /tmp/utmos_orig
    python benchmark_accuracy.py --audio_dir Elise_audio --num_files 100 \
        --out orig.csv --use_upstream /tmp/utmos_orig

Step 3 — compare:
    python benchmark_accuracy.py --compare ours.csv orig.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from tqdm import tqdm
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")

    # ── predict ──────────────────────────────────────────────────────────────
    pred = sub.add_parser("predict", help="Run predictions and save to CSV")
    pred.add_argument("--audio_dir", required=True)
    pred.add_argument("--num_files", type=int, default=None)
    pred.add_argument("--num_repetitions", type=int, default=5,
                      help="Higher = more stable (default 5 for accuracy test)")
    pred.add_argument("--out", required=True, help="Output CSV path")
    pred.add_argument("--use_upstream", default=None,
                      help="Path where upstream UTMOSv2 is installed (--target dir). "
                           "If given, import from there instead of this fork.")

    # ── compare ──────────────────────────────────────────────────────────────
    cmp = sub.add_parser("compare", help="Compare two prediction CSVs")
    cmp.add_argument("a", help="First CSV (e.g. ours.csv)")
    cmp.add_argument("b", help="Second CSV (e.g. orig.csv)")
    cmp.add_argument("--label_a", default="ours")
    cmp.add_argument("--label_b", default="original")

    # ── legacy flat usage (no subcommand) ────────────────────────────────────
    p.add_argument("--audio_dir")
    p.add_argument("--num_files", type=int, default=None)
    p.add_argument("--num_repetitions", type=int, default=5)
    p.add_argument("--out")
    p.add_argument("--use_upstream", default=None)
    p.add_argument("--compare", nargs=2, metavar=("A", "B"))

    return p.parse_args()


# ── helpers ──────────────────────────────────────────────────────────────────

def load_csv(path: str) -> dict[str, float]:
    with open(path) as f:
        reader = csv.DictReader(f)
        return {Path(row["file_path"]).stem: float(row["predicted_mos"]) for row in reader}


def save_csv(path: str, results: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file_path", "predicted_mos"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved {len(results)} predictions → {path}")


def run_compare(path_a: str, path_b: str,
                label_a: str = "A", label_b: str = "B") -> None:
    a = load_csv(path_a)
    b = load_csv(path_b)
    common = sorted(set(a) & set(b))
    if not common:
        print("ERROR: no common files between the two CSVs.")
        sys.exit(1)

    va = np.array([a[k] for k in common])
    vb = np.array([b[k] for k in common])
    diff = va - vb

    corr = float(np.corrcoef(va, vb)[0, 1])
    mae  = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    bias = float(np.mean(diff))

    print(f"\nComparing {label_a}  vs  {label_b}")
    print(f"  Common files   : {len(common)}")
    print(f"  Pearson r      : {corr:.6f}  (1.000 = perfect agreement)")
    print(f"  MAE            : {mae:.6f}")
    print(f"  RMSE           : {rmse:.6f}")
    print(f"  Mean bias      : {bias:+.6f}  ({label_a} minus {label_b})")

    if corr > 0.9999 and mae < 0.01:
        print("\n  ✓ Predictions are numerically equivalent.")
    elif corr > 0.999:
        print("\n  ~ Very high correlation; minor numerical differences only.")
    elif corr > 0.99:
        print("\n  ~ High correlation; small systematic differences present.")
    else:
        print("\n  ✗ Notable differences — check which changes affect predictions.")

    # Top-10 largest discrepancies
    order = np.argsort(np.abs(diff))[::-1][:10]
    print(f"\n  Largest discrepancies (top 10):")
    print(f"  {'File':<50} {label_a:>8} {label_b:>8}  {'Δ':>8}")
    for i in order:
        k = common[i]
        print(f"  {k[:50]:<50} {va[i]:>8.4f} {vb[i]:>8.4f}  {diff[i]:>+8.4f}")


def run_predict(args) -> None:
    if args.use_upstream:
        sys.path.insert(0, args.use_upstream)
        print(f"Using upstream UTMOSv2 from: {args.use_upstream}")

    import utmosv2  # noqa: E402 — imported after optional path tweak

    audio_dir = Path(args.audio_dir)
    files = sorted(list(audio_dir.glob("*.wav")) + list(audio_dir.glob("*.mp3")))
    if args.num_files is not None:
        files = files[: args.num_files]
    assert files, f"No audio files in {audio_dir}"

    print(f"Files           : {len(files)}")
    print(f"num_repetitions : {args.num_repetitions}")
    print("Loading model...")
    model = utmosv2.create_model()
    print()

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = []
    for f in tqdm(files):
        score = model.predict(
            input_path=f,
            device=device,
            num_repetitions=args.num_repetitions,
            verbose=False,
        )
        results.append({"file_path": str(f), "predicted_mos": score})

    save_csv(args.out, results)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # subcommand style
    if args.cmd == "predict":
        run_predict(args)
    elif args.cmd == "compare":
        run_compare(args.a, args.b, args.label_a, args.label_b)
    # legacy flat style
    elif args.compare:
        run_compare(args.compare[0], args.compare[1])
    elif args.out:
        run_predict(args)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
