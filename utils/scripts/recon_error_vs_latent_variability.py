#!/usr/bin/env python3
"""
Per-patient reconstruction error vs latent variability (from slice embeddings NPZ).

StageB NPZ (plain_ae / slice_meta) typically has ``embeddings`` only — no image recon loss.
You must supply a table with one reconstruction metric per patient, e.g. from
``eval_diffae_recon.py`` volume CSV (``mse_mean``) or your own AE eval.

Variability scalars (after trajectory ordering: mask, slice_z_mm lexsort — same as vol_mm pipeline):
  - mean_std: mean_j std_t z[t,j]
  - mean_p90_p10: mean_j (P90-P10)_t along slices
  - mean_vol_mm: mean_j vol_mm_j (needs mm; slice_z_mm)
  - l2_std: || std_vector ||_2

Outputs
-------
- out_dir/patient_table.csv — patient_id, recon_error, variability_scalar, optional label
- out_dir/summary.json — n_patients, pearson_r, pearson_p, spearman_r, spearman_p (if scipy)
- out_dir/scatter_recon_vs_var.png

Example
-------
  python recon_error_vs_latent_variability.py \\
    --recon_csv /path/volume_metrics.csv \\
    --patient_id_col patient_id \\
    --recon_col mse_mean \\
    --npz_dir /path/diff3dformer_stageB_plain_ae_slice_meta/patients \\
    --variability mean_std \\
    --labels_csv data/train.csv --labels_csv data/val.csv --labels_csv data/test.csv \\
    --out_dir ./recon_vs_var_out
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd

_scripts_dir = os.path.dirname(os.path.abspath(__file__))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from repeated_stratified_shuffle_eval import _merge_labels_and_clinical, _sex_to_bin  # noqa: E402
import trajectory_volatility_classifier_cv as traj_cv  # noqa: E402

try:
    from scipy.stats import pearsonr, spearmanr
except ImportError:
    pearsonr = spearmanr = None  # type: ignore

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # type: ignore


def _scalar_variability(emb: np.ndarray, zmm: np.ndarray, eps_mm: float, kind: str) -> float:
    fd = traj_cv._per_patient_feats(emb, zmm, eps_mm)
    if kind == "mean_std":
        v = fd["std"]
    elif kind == "mean_p90_p10":
        v = fd["p90_p10"]
    elif kind == "mean_vol_mm":
        v = fd["vol_mm"]
    elif kind == "l2_std":
        return float(np.linalg.norm(fd["std"], ord=2))
    else:
        raise ValueError(kind)
    return float(np.mean(v))


def main() -> None:
    ap = argparse.ArgumentParser(description="Recon error vs latent variability per patient.")
    ap.add_argument("--recon_csv", required=True, help="CSV with patient_id and recon column.")
    ap.add_argument("--patient_id_col", default="patient_id")
    ap.add_argument("--recon_col", required=True, help="Numeric reconstruction error column (e.g. mse_mean).")
    ap.add_argument("--npz_dir", required=True)
    ap.add_argument(
        "--variability",
        choices=["mean_std", "mean_p90_p10", "mean_vol_mm", "l2_std"],
        default="mean_std",
    )
    ap.add_argument("--eps_mm", type=float, default=1e-3)
    ap.add_argument("--labels_csv", action="append", default=None, help="Optional: for label column in table / hue.")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    rdf = pd.read_csv(args.recon_csv)
    idc = str(args.patient_id_col)
    rc = str(args.recon_col)
    if idc not in rdf.columns or rc not in rdf.columns:
        raise SystemExit(f"{args.recon_csv} needs columns {idc!r} and {rc!r}, got {list(rdf.columns)}")

    rdf = rdf[[idc, rc]].copy()
    rdf[idc] = rdf[idc].astype(str)
    rdf = rdf.dropna(subset=[rc])
    rdf = rdf.groupby(idc, as_index=False)[rc].mean()

    npz_dir = os.path.abspath(args.npz_dir)
    y_by_pid: Optional[dict[str, int]] = None
    if args.labels_csv:
        clin = _merge_labels_and_clinical([os.path.abspath(p) for p in args.labels_csv])
        clin = clin.set_index("patient_id", drop=False)
        y_by_pid = {}
        for pid in clin.index.astype(str):
            row = clin.loc[pid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            if pd.isna(row.get("age")):
                continue
            sb = _sex_to_bin(row["sex"])
            if not np.isfinite(sb):
                continue
            y_by_pid[str(pid)] = int(row["label"])

    rows: list[dict] = []
    for fn in sorted(os.listdir(npz_dir)):
        if not fn.endswith(".npz"):
            continue
        pid = fn.replace(".npz", "")
        sub = rdf[rdf[idc] == pid]
        if sub.empty:
            continue
        err = float(sub[rc].iloc[0])
        path = os.path.join(npz_dir, fn)
        try:
            z = np.load(path, allow_pickle=True)
            emb, zmm, _ = traj_cv._ordered_rows(z)
        except (ValueError, KeyError, OSError):
            continue
        if emb.shape[0] < 2:
            continue
        try:
            var_s = _scalar_variability(emb, zmm, float(args.eps_mm), str(args.variability))
        except Exception:
            continue
        if not np.isfinite(var_s) or not np.isfinite(err):
            continue
        rec: dict = {"patient_id": pid, "recon_error": err, "variability": var_s, "variability_kind": args.variability}
        if y_by_pid and pid in y_by_pid:
            rec["label"] = y_by_pid[pid]
        rows.append(rec)

    if len(rows) < 5:
        raise SystemExit(f"Too few matched patients ({len(rows)}). Check IDs in recon_csv vs NPZ stems.")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "patient_table.csv"), index=False)

    x = df["recon_error"].to_numpy(dtype=np.float64)
    y = df["variability"].to_numpy(dtype=np.float64)
    summ: dict = {"n_patients": int(len(df)), "variability": str(args.variability), "recon_col": rc}
    if pearsonr is not None:
        pr, pp = pearsonr(x, y)
        sr, sp = spearmanr(x, y)
        summ["pearson_r"] = float(pr)
        summ["pearson_p"] = float(pp)
        summ["spearman_r"] = float(sr)
        summ["spearman_p_sp"] = float(sp)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summ, f, indent=2)
    print(json.dumps(summ, indent=2))

    if plt is not None:
        fig, ax = plt.subplots(figsize=(6, 5))
        if "label" in df.columns:
            for lab, g in df.groupby("label"):
                ax.scatter(g["recon_error"], g["variability"], alpha=0.65, s=28, label=f"label={lab}")
            ax.legend()
        else:
            ax.scatter(x, y, alpha=0.65, s=28)
        ax.set_xlabel(f"Reconstruction error ({rc})")
        ax.set_ylabel(f"Latent variability ({args.variability}; trajectory-ordered slices)")
        ax.set_title("Per-patient recon error vs latent variability")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "scatter_recon_vs_var.png"), dpi=150)
        plt.close(fig)
    else:
        print("[warn] matplotlib not available; skip scatter png", file=sys.stderr)

    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
