"""Stage 4: aux features, stratified 80/10/10 split, scaler (train-fit only).
Outputs (cfg.splits_dir): splits.json, aux_cols.json, scaler.npz,
                          aux_medians.json, table.csv
Run: python -m src.preprocessing
"""
from __future__ import annotations
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import get_config
from .utils import setup_logging


def main():
    cfg = get_config("Stage 4: splits + scaling")
    cfg.ensure_dirs()
    log = setup_logging(cfg, "preprocessing")

    df = pd.read_csv(cfg.processed_csv)
    graphs = torch.load(cfg.graph_cache, weights_only=False)
    with open(cfg.natoms_map) as f:
        natoms = json.load(f)

    df = df[df["id"].isin(graphs)].reset_index(drop=True)

    # repair natoms / energy_per_atom with CIF-derived atom counts
    nat = df["id"].map(lambda i: natoms.get(i, np.nan)).astype(float)
    df["natoms_formula"] = df["natoms_formula"].fillna(nat)
    df["energy_per_atom"] = df["energy_per_atom"].fillna(df["energy_total"] / nat)

    aux_cols = (["pld", "lcd", "density", "pld_missing", "lcd_missing"]
                + (["energy_per_atom"] if cfg.use_energy else []))
    X = df[aux_cols].values.astype(np.float32)
    X[~np.isfinite(X)] = np.nan
    y = df["bandgap"].values.astype(np.float32)

    # stratify on gap deciles -> balanced splits despite the peak at 0 eV
    bins = pd.qcut(y, q=10, labels=False, duplicates="drop")
    idx = np.arange(len(df))
    tr_tmp, te_idx = train_test_split(idx, test_size=0.10, stratify=bins,
                                      random_state=cfg.seed)
    tr_idx, va_idx = train_test_split(tr_tmp, test_size=0.111, stratify=bins[tr_tmp],
                                      random_state=cfg.seed)

    # impute residual NaNs (density/energy) with TRAIN medians -> no leakage
    medians = {}
    for j, c in enumerate(aux_cols):
        med = (float(np.nanmedian(X[tr_idx, j]))
               if np.isfinite(X[tr_idx, j]).any() else 0.0)
        medians[c] = med
        nan_mask = np.isnan(X[:, j])
        if nan_mask.any():
            X[nan_mask, j] = med
            log.info("imputed %d NaNs in %s with train median %.4f",
                     int(nan_mask.sum()), c, med)

    scaler = StandardScaler().fit(X[tr_idx])          # fit on TRAIN only
    Xs = scaler.transform(X).astype(np.float32)

    os.makedirs(cfg.splits_dir, exist_ok=True)
    ids = df["id"].tolist()
    with open(os.path.join(cfg.splits_dir, "splits.json"), "w") as f:
        json.dump({"train": [ids[i] for i in tr_idx],
                   "val": [ids[i] for i in va_idx],
                   "test": [ids[i] for i in te_idx]}, f)
    with open(os.path.join(cfg.splits_dir, "aux_cols.json"), "w") as f:
        json.dump({"aux_cols": aux_cols, "use_aux": cfg.use_aux}, f)
    np.savez(os.path.join(cfg.splits_dir, "scaler.npz"),
             mean=scaler.mean_, scale=scaler.scale_)
    with open(os.path.join(cfg.splits_dir, "aux_medians.json"), "w") as f:
        json.dump(medians, f, indent=2)

    keep = ["id", "formula", "bandgap", "pld", "lcd", "density",
            "energy_total", "energy_per_atom", "natoms_formula",
            "pld_missing", "lcd_missing"]
    df[keep].to_csv(os.path.join(cfg.splits_dir, "table.csv"), index=False)

    log.info("aux dims %d | train/val/test = %d/%d/%d",
             len(aux_cols), len(tr_idx), len(va_idx), len(te_idx))


if __name__ == "__main__":
    main()
