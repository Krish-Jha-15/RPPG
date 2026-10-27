"""Test signal-only temporal consistency for camera rPPG spectral candidates.

Examples (run from the RPPG project folder):
    python test_temporally_consistent_hr.py --video 6 --ground-truth gtdump_6.xmp
    python test_temporally_consistent_hr.py --video 7 --ground-truth gtdump_7.xmp

Candidate selection uses relative spectral power and changes between adjacent
windows only. Ground-truth HR is used only after selection for evaluation.
This is an experiment, not a replacement for the existing baseline pipeline.
"""
from pathlib import Path
import argparse
import re
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path("results")
SUMMARY_SUFFIX = "_spectral_candidates_summary.csv"
# Fixed settings. Do not tune these against the evaluation ground truth.
SMOOTHNESS_WEIGHT = 0.8
DELTA_SCALE_BPM = 10.0
MIN_RELATIVE_POWER = 1e-6


def find_latest_summary(video_num: int) -> Path:
    if not RESULTS_DIR.is_dir():
        raise FileNotFoundError(
            f"Cannot find {RESULTS_DIR.resolve()}. Run from the project folder containing 'results'."
        )
    pat = re.compile(rf"(?:vid|video)[_-]?{video_num}(?:[_-]|$)", re.IGNORECASE)
    paths = [p for p in RESULTS_DIR.rglob(f"video{video_num}{SUMMARY_SUFFIX}") if pat.search(str(p))]
    if not paths:
        paths = [p for p in RESULTS_DIR.rglob("*spectral_candidates_summary.csv") if pat.search(str(p))]
    if not paths:
        raise FileNotFoundError(
            f"No spectral-candidate summary found for Video {video_num}. "
            f"Run inspect_camera_spectral_candidates.py --video {video_num} first."
        )
    return max(paths, key=lambda p: p.stat().st_mtime)


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    a, b = y_true[mask], y_pred[mask]
    if not len(a):
        return {"Windows": 0, "MAE_BPM": np.nan, "RMSE_BPM": np.nan,
                "Bias_BPM": np.nan, "Within_5_BPM_pct": np.nan}
    err = b - a
    return {
        "Windows": int(len(a)),
        "MAE_BPM": float(np.mean(np.abs(err))),
        "RMSE_BPM": float(np.sqrt(np.mean(err ** 2))),
        "Bias_BPM": float(np.mean(err)),
        "Within_5_BPM_pct": float(np.mean(np.abs(err) <= 5.0) * 100.0),
    }


def get_candidate_columns(df: pd.DataFrame):
    ranks = []
    for col in df.columns:
        m = re.fullmatch(r"Candidate_(\d+)_HR_BPM", col)
        if m:
            rank = int(m.group(1))
            pcol = f"Candidate_{rank}_Power_Ratio"
            if pcol in df.columns:
                ranks.append((rank, col, pcol))
    return sorted(ranks)


def select_path(df: pd.DataFrame, candidate_cols):
    """Viterbi-style candidate selection using spectral power + smoothness only."""
    candidate_rows = []
    emission_costs = []
    for _, row in df.iterrows():
        cands, costs = [], []
        for rank, hr_col, power_col in candidate_cols:
            hr = pd.to_numeric(pd.Series([row.get(hr_col, np.nan)]), errors="coerce").iloc[0]
            ratio = pd.to_numeric(pd.Series([row.get(power_col, np.nan)]), errors="coerce").iloc[0]
            if not np.isfinite(hr) or not np.isfinite(ratio) or ratio <= 0:
                continue
            ratio = float(np.clip(ratio, MIN_RELATIVE_POWER, 1.0))
            cands.append({"rank": rank, "hr": float(hr), "power_ratio": ratio})
            costs.append(float(-np.log(ratio)))
        candidate_rows.append(cands)
        emission_costs.append(costs)

    if not candidate_rows or any(len(x) == 0 for x in candidate_rows):
        raise ValueError("At least one window has no usable spectral candidates.")

    dp = [np.asarray(emission_costs[0], dtype=float)]
    backptr = [np.full(len(candidate_rows[0]), -1, dtype=int)]
    for i in range(1, len(candidate_rows)):
        prev, curr = candidate_rows[i - 1], candidate_rows[i]
        current_cost = np.empty(len(curr), dtype=float)
        current_back = np.empty(len(curr), dtype=int)
        for j, c in enumerate(curr):
            transitions = np.array([
                dp[i - 1][k] + SMOOTHNESS_WEIGHT * ((c["hr"] - p["hr"]) / DELTA_SCALE_BPM) ** 2
                for k, p in enumerate(prev)
            ])
            best_k = int(np.argmin(transitions))
            current_cost[j] = transitions[best_k] + emission_costs[i][j]
            current_back[j] = best_k
        dp.append(current_cost)
        backptr.append(current_back)

    state = int(np.argmin(dp[-1]))
    selected = [None] * len(candidate_rows)
    for i in range(len(candidate_rows) - 1, -1, -1):
        selected[i] = candidate_rows[i][state]
        if i > 0:
            state = int(backptr[i][state])
    return selected


def find_gt_file(summary_path: Path, video_num: int, explicit_path: str | None = None) -> Path | None:
    """Resolve matching GT file. Explicit --ground-truth takes priority."""
    if explicit_path:
        p = Path(explicit_path).expanduser()
        if p.is_file():
            return p.resolve()
        raise FileNotFoundError(f"The --ground-truth path does not exist: {p}")

    result_folder = summary_path.parent.parent
    candidates: list[Path] = []
    run_summary = result_folder / "run_summary.txt"
    if run_summary.is_file():
        txt = run_summary.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"(?mi)^Ground\s*truth:\s*(.+?)\s*$", txt)
        if match:
            raw = match.group(1).strip().strip('"')
            raw_path = Path(raw)
            candidates.append(raw_path)
            if not raw_path.is_absolute():
                candidates.extend([result_folder / raw_path, Path.cwd() / raw_path,
                                   result_folder.parent.parent / raw_path])
            candidates.extend([Path.cwd() / raw_path.name, result_folder.parent.parent / raw_path.name])
    candidates.extend([
        Path.cwd() / f"gtdump_{video_num}.xmp",
        result_folder.parent.parent / f"gtdump_{video_num}.xmp",
    ])
    seen = set()
    for item in candidates:
        try:
            key = str(item.resolve()).lower()
        except Exception:
            key = str(item).lower()
        if key in seen:
            continue
        seen.add(key)
        if item.is_file():
            return item.resolve()
    return None


def read_gt_timestamp_hr(gt_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read timestamp/HR from comma-, tab-, space-, or XML-like XMP text.

    UBFC gtdump files usually contain timestamp_ms, HR, SpO2, PPG per row.
    A line-oriented numeric fallback avoids separator/header ambiguity.
    """
    rows: list[tuple[float, float]] = []
    numeric_re = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
    text = gt_path.read_text(encoding="utf-8-sig", errors="replace")
    for line in text.splitlines():
        nums = numeric_re.findall(line)
        if len(nums) < 2:
            continue
        try:
            timestamp = float(nums[0])
            hr = float(nums[1])
        except ValueError:
            continue
        # Filter non-data numeric lines and headers. Physiological HR is expected in this broad range.
        if np.isfinite(timestamp) and np.isfinite(hr) and timestamp >= 0 and 25 <= hr <= 240:
            rows.append((timestamp, hr))

    if len(rows) < 10:
        # Fallback through pandas separator detection, if line-based parsing did not find enough rows.
        try:
            table = pd.read_csv(gt_path, sep=None, engine="python", header=None, comment="#", on_bad_lines="skip")
        except Exception:
            table = pd.DataFrame()
        if table.shape[1] == 1:
            try:
                table = pd.read_csv(gt_path, sep=r"[,;\s]+", engine="python", header=None, comment="#", on_bad_lines="skip")
            except Exception:
                pass
        alt_rows = []
        for _, row in table.iterrows():
            vals = pd.to_numeric(row, errors="coerce").to_numpy(float)
            vals = vals[np.isfinite(vals)]
            if len(vals) >= 2 and vals[0] >= 0 and 25 <= vals[1] <= 240:
                alt_rows.append((float(vals[0]), float(vals[1])))
        if len(alt_rows) > len(rows):
            rows = alt_rows

    if len(rows) < 10:
        raise ValueError(
            f"Could not parse enough timestamp/HR rows from {gt_path}. Found {len(rows)} plausible rows."
        )

    arr = np.asarray(rows, dtype=float)
    times_raw, hrs = arr[:, 0], arr[:, 1]
    # UBFC gtdump timestamps are in milliseconds (e.g. ending around 80,000).
    times = times_raw / 1000.0 if np.nanmax(times_raw) > 1000 else times_raw
    order = np.argsort(times)
    times, hrs = times[order], hrs[order]
    # Remove duplicate timestamps if any.
    unique_times, unique_idx = np.unique(times, return_index=True)
    return unique_times, hrs[unique_idx]


def fill_missing_gt_window_means(df: pd.DataFrame, summary_path: Path, video_num: int,
                                 explicit_gt_path: str | None = None) -> tuple[pd.DataFrame, Path]:
    """Use gtdump only for evaluation; it never affects selected spectral peaks."""
    out = df.copy()
    if "GT_HR_Mean_BPM" not in out.columns:
        out["GT_HR_Mean_BPM"] = np.nan
    existing = pd.to_numeric(out["GT_HR_Mean_BPM"], errors="coerce").to_numpy(dtype=float, copy=True)
    missing = ~np.isfinite(existing)
    if not missing.any():
        return out, Path("<embedded-in-candidate-table>")

    gt_path = find_gt_file(summary_path, video_num, explicit_gt_path)
    if gt_path is None:
        raise FileNotFoundError(
            f"Ground-truth HR is missing from the candidate table and no gtdump file was found. "
            f"Run with --ground-truth gtdump_{video_num}.xmp (or a full path)."
        )
    times, hrs = read_gt_timestamp_hr(gt_path)
    print(f"Ground-truth fallback for evaluation only: {gt_path}")
    for idx in np.flatnonzero(missing):
        start = float(out.iloc[idx]["Start_sec"])
        end = float(out.iloc[idx]["End_sec"])
        mask = (times >= start) & (times < end)
        if mask.any():
            existing[idx] = float(np.mean(hrs[mask]))
    out["GT_HR_Mean_BPM"] = existing
    if not np.isfinite(existing).all():
        bad = out.loc[~np.isfinite(existing), ["Start_sec", "End_sec"]].to_dict("records")
        raise ValueError(f"Could not compute ground-truth means for all windows. Unresolved windows: {bad}")
    return out, gt_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=int, choices=list(range(1, 8)), default=4,
                        help="Recording number to analyze (1-7; default: 4)")
    parser.add_argument("--ground-truth", type=str, default=None,
                        help="Explicit path to matching gtdump_X.xmp; useful if auto-discovery fails")
    args = parser.parse_args()

    summary_path = find_latest_summary(args.video)
    df = pd.read_csv(summary_path)
    df, gt_path = fill_missing_gt_window_means(df, summary_path, args.video, args.ground_truth)
    needed = {"Start_sec", "End_sec", "GT_HR_Mean_BPM", "Candidate_1_HR_BPM", "Candidate_1_Power_Ratio"}
    missing = sorted(needed - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {summary_path}: {missing}")
    candidate_cols = get_candidate_columns(df)
    if len(candidate_cols) < 2:
        raise ValueError(f"Only {len(candidate_cols)} candidate rank(s) found; at least 2 required.")

    selected = select_path(df, candidate_cols)
    out = df[["Start_sec", "End_sec", "GT_HR_Mean_BPM", "Candidate_1_HR_BPM"]].copy()
    out = out.rename(columns={"Candidate_1_HR_BPM": "Dominant_Peak_HR_BPM"})
    out["Temporally_Selected_HR_BPM"] = [c["hr"] for c in selected]
    out["Selected_Candidate_Rank"] = [c["rank"] for c in selected]
    out["Selected_Power_Ratio"] = [c["power_ratio"] for c in selected]
    out["Delta_from_Dominant_BPM"] = out["Temporally_Selected_HR_BPM"] - out["Dominant_Peak_HR_BPM"]
    out["Dominant_Abs_Error_BPM_Evaluation_Only"] = np.abs(out["Dominant_Peak_HR_BPM"] - out["GT_HR_Mean_BPM"])
    out["Selected_Abs_Error_BPM_Evaluation_Only"] = np.abs(out["Temporally_Selected_HR_BPM"] - out["GT_HR_Mean_BPM"])

    out_dir = summary_path.parent / "temporal_peak_selection_test"
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / f"video{args.video}_temporal_peak_selection.csv"
    metrics_path = out_dir / f"video{args.video}_temporal_peak_metrics.csv"
    plot_path = out_dir / f"video{args.video}_temporal_peak_selection.png"
    out.to_csv(csv_path, index=False, float_format="%.4f")

    y = pd.to_numeric(out["GT_HR_Mean_BPM"], errors="coerce").to_numpy(float)
    dominant = pd.to_numeric(out["Dominant_Peak_HR_BPM"], errors="coerce").to_numpy(float)
    chosen = pd.to_numeric(out["Temporally_Selected_HR_BPM"], errors="coerce").to_numpy(float)
    metrics = pd.DataFrame([
        {"Method": name, **metric_dict(y, pred)}
        for name, pred in [("Dominant peak baseline", dominant), ("Temporal continuity selection", chosen)]
    ])
    metrics.to_csv(metrics_path, index=False, float_format="%.4f")

    centers = (out["Start_sec"].to_numpy(float) + out["End_sec"].to_numpy(float)) / 2.0
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    ax.plot(centers, dominant, marker="o", linewidth=1.5, label="Strongest FFT peak (baseline)")
    ax.plot(centers, chosen, marker="s", linewidth=1.5, label="Signal-only temporal selection")
    ax.plot(centers, y, marker=".", linewidth=1.5, linestyle="--", label="Supplied GT HR (evaluation only)")
    ax.set_xlabel("Window center time (s)")
    ax.set_ylabel("HR (BPM)")
    ax.set_title(f"Video {args.video}: dominant peak vs temporal candidate selection")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    print(f"VIDEO {args.video} TEMPORAL SPECTRAL-CANDIDATE SELECTION TEST")
    print(f"Input spectral candidate table: {summary_path.resolve()}")
    print(f"Ground-truth file used for evaluation: {gt_path}")
    print("Selection uses candidate relative power + temporal smoothness only; GT is excluded from selection.")
    print(f"Fixed smoothness weight: {SMOOTHNESS_WEIGHT:.3f}; HR change scale: {DELTA_SCALE_BPM:.1f} BPM per window hop")
    print(f"Windows analyzed: {len(out)}; candidate ranks available: {len(candidate_cols)}")
    print(f"Saved selected track: {csv_path.resolve()}")
    print(f"Saved metrics: {metrics_path.resolve()}")
    print(f"Saved plot: {plot_path.resolve()}")
    print("\nEvaluation metrics (GT used only after selection):")
    print(metrics.round(3).to_string(index=False))
    show_cols = ["Start_sec", "End_sec", "Dominant_Peak_HR_BPM", "Temporally_Selected_HR_BPM",
                 "Selected_Candidate_Rank", "Selected_Power_Ratio", "GT_HR_Mean_BPM"]
    print("\nSelected candidate by window:")
    print(out[show_cols].round(3).to_string(index=False))
    print("\nCaution: this exploratory temporal prior can oversmooth real HR changes or follow a consistent wrong peak.")
    print("Do not replace the baseline until results are checked across multiple recordings.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
