"""Parse Dataset 2 ground_truth.txt (three whitespace-separated numeric lines).

Expected logical lines:
  1) PPG signal samples
  2) HR values in BPM
  3) timestamps in seconds (scientific notation is supported)

The parser intentionally does not guess how to align arrays if their lengths differ.
It preserves the supplied samples and timestamps; it does not resample or smooth them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd


def parse_numeric_line(line: str, line_number: int) -> np.ndarray:
    """Parse one whitespace-delimited line of floats, including scientific notation."""
    # np.fromstring is robust to spaces/tabs and values such as 1.2028675e-01.
    values = np.fromstring(line.strip(), sep=" ", dtype=float)
    if values.size == 0:
        raise ValueError(f"Line {line_number} contains no numeric values.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Line {line_number} contains NaN or infinite values.")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parse Dataset 2's 3-line ground_truth.txt to a validated CSV."
    )
    parser.add_argument("ground_truth", help="Path to Dataset 2 ground_truth.txt")
    parser.add_argument(
        "--output",
        help="Output CSV path (default: <input-folder>/ground_truth_parsed.csv)",
    )
    args = parser.parse_args()

    src = Path(args.ground_truth).expanduser().resolve()
    if not src.is_file():
        print(f"ERROR: Ground-truth file not found: {src}", file=sys.stderr)
        return 2

    try:
        raw_text = src.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        raw_text = src.read_text(encoding="latin-1")

    # Dataset documentation describes exactly three logical lines. Do not silently
    # concatenate arbitrary lines because that could hide a format mismatch.
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    if len(lines) != 3:
        print(
            "ERROR: Expected exactly 3 non-empty lines (PPG, HR, timestamps), "
            f"but found {len(lines)}.\n"
            "Check whether the file has line wrapping, labels, or additional headers.",
            file=sys.stderr,
        )
        return 2

    try:
        ppg = parse_numeric_line(lines[0], 1)
        hr = parse_numeric_line(lines[1], 2)
        timestamps = parse_numeric_line(lines[2], 3)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    lengths = {"PPG": len(ppg), "HR": len(hr), "timestamps": len(timestamps)}
    print("DATASET 2 GROUND-TRUTH CHECK")
    print(f"Input: {src}")
    print("Parsed sample counts:")
    for name, count in lengths.items():
        print(f"  {name}: {count}")

    if len(set(lengths.values())) != 1:
        print(
            "\nERROR: PPG, HR, and timestamp counts differ. No CSV was written because "
            "aligning the values by truncation or guessing could corrupt evaluation.",
            file=sys.stderr,
        )
        return 2

    diffs = np.diff(timestamps)
    if np.any(diffs < 0):
        bad_count = int(np.count_nonzero(diffs < 0))
        print(
            f"ERROR: Timestamps are not nondecreasing ({bad_count} backward steps). "
            "No CSV was written.",
            file=sys.stderr,
        )
        return 2

    duration = float(timestamps[-1] - timestamps[0]) if timestamps.size > 1 else 0.0
    duplicate_count = int(np.count_nonzero(diffs == 0))
    positive_diffs = diffs[diffs > 0]
    print(f"Timestamp start: {timestamps[0]:.6f} s")
    print(f"Timestamp end:   {timestamps[-1]:.6f} s")
    print(f"Duration:        {duration:.6f} s")
    print(f"Duplicate timestamp steps: {duplicate_count}")
    if positive_diffs.size:
        median_dt = float(np.median(positive_diffs))
        print(f"Median positive timestamp interval: {median_dt:.6f} s")
        print(f"Approx. rate from median interval: {1.0 / median_dt:.3f} Hz")
        print(
            "Positive interval range: "
            f"{float(positive_diffs.min()):.6f} to {float(positive_diffs.max()):.6f} s"
        )

    out = Path(args.output).expanduser().resolve() if args.output else src.parent / "ground_truth_parsed.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "Timestamp_sec": timestamps,
            "HR_BPM": hr,
            "PPG": ppg,
        }
    )
    df.to_csv(out, index=False, float_format="%.10g")

    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))
    print(f"\nSaved validated CSV: {out}")
    print(
        "NOTE: This step only parses and validates. It preserves duplicate/irregular "
        "timestamps and does not resample PPG or change the existing rPPG pipeline."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
