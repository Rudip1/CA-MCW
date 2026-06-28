"""log_writer.py — persist training history as CSV + PNG curves.

Both train_inference.py and train_imitation.py call save_training_log()
after the trainer returns. Writes two files into out_dir:

    train_log_<kind>_<run_id>.csv     # one row per epoch, all history keys
    loss_curve_<kind>_<run_id>.png    # train + val loss vs epoch

Without these, the only training record was stdout — useless for
multi-seed analysis (plan.md Phase E11.1) and for figure work.
"""
from __future__ import annotations

import csv
import os
import time
from typing import Dict, List, Optional, Tuple


def _resolve_run_id(run_id: Optional[str]) -> str:
    if run_id:
        return str(run_id)
    return time.strftime("%Y%m%d_%H%M%S")


def _write_csv(history: List[Dict[str, float]], path: str) -> None:
    if not history:
        return
    keys = list(history[0].keys())
    for r in history[1:]:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in history:
            w.writerow({k: row.get(k, "") for k in keys})


def _write_png(history: List[Dict[str, float]], path: str, title: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"[log_writer] matplotlib unavailable; skipping {path}")
        return

    epochs = [r["epoch"] for r in history]
    train = [r.get("train_loss", float("nan")) for r in history]
    val = [r.get("val_loss", float("nan")) for r in history]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, train, label="train", marker="o", markersize=3)
    ax.plot(epochs, val, label="val", marker="s", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_training_log(
    history: List[Dict[str, float]],
    out_dir: str,
    kind: str,
    run_id: Optional[str] = None,
) -> Tuple[str, str]:
    """Persist training history. Returns (csv_path, png_path).

    Args:
      history:  trainer's result.history list (one dict per epoch).
      out_dir:  run folder (e.g. models/metacritic_raw_wt/raw_run1_2026_05_10/).
                The folder name already identifies the run, so filenames
                are fixed: training_log.csv and training_curve.png.
      kind:     "inference" or "imitation"; used only in the plot title.
      run_id:   free-form id stamped into the plot title for reference.
    """
    rid = _resolve_run_id(run_id)
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "training_log.csv")
    png_path = os.path.join(out_dir, "training_curve.png")

    _write_csv(history, csv_path)
    _write_png(history, png_path, f"{kind} training ({os.path.basename(out_dir)})")

    print(f"[log_writer] wrote {csv_path}")
    print(f"[log_writer] wrote {png_path}")
    return csv_path, png_path
