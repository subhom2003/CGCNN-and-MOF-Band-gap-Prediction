"""Stage 1: load qmof.csv, select/rename columns, clean, derive features.
Output: data/processed/qmof_clean.csv
Run:    python -m src.data_loading
"""
from __future__ import annotations
import os

import numpy as np
import pandas as pd

from .config import get_config
from .utils import COLS, comp_dict, setup_logging


def load_and_clean(cfg):
    raw = pd.read_csv(cfg.raw_csv)
    missing = [c for c in COLS if c not in raw.columns]
    if missing:
        raise KeyError(f"columns missing from raw csv: {missing}")
    df = raw[list(COLS)].rename(columns=COLS).copy()
    n_raw = len(df)

    # IDs -> CIF paths
    df["id"] = df["id"].astype(str).str.strip()
    df["cif_path"] = df["id"].map(lambda x: os.path.join(cfg.cif_dir, f"{x}.cif"))
    df["has_cif"] = df["cif_path"].map(os.path.exists)

    # Target: numeric, drop NaN, clip tiny DFT negative artifacts to 0
    df["bandgap"] = pd.to_numeric(df["bandgap"], errors="coerce")
    df = df.dropna(subset=["bandgap"])
    df["bandgap"] = df["bandgap"].clip(lower=0.0)

    # Auxiliary numerics
    for c in ["pld", "lcd", "density", "energy_total"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # composition -> natoms -> energy/atom (total energy is extensive)
    df["formula"] = df["formula"].astype(str).str.strip()
    df["comp_dict"] = df["formula"].map(comp_dict)
    df["natoms_formula"] = df["comp_dict"].map(lambda d: sum(d.values()) if d else np.nan)
    df["energy_per_atom"] = df["energy_total"] / df["natoms_formula"]
    df["energy_per_atom"] = df["energy_per_atom"].replace([np.inf, -np.inf], np.nan)

    # missing pore descriptors => dense/non-porous -> 0 + indicator flag
    df["pld_missing"] = df["pld"].isna().astype(np.float32)
    df["lcd_missing"] = df["lcd"].isna().astype(np.float32)
    df["pld"] = df["pld"].fillna(0.0)
    df["lcd"] = df["lcd"].fillna(0.0)

    # exact duplicates (same composition & geometry & energy)
    dup = df.duplicated(subset=["formula", "pld", "lcd", "density", "energy_total"], keep="first")
    df = df[~dup].reset_index(drop=True)

    stats = {"n_raw": int(n_raw), "n_after_clean": int(len(df)),
             "n_with_cif": int(df["has_cif"].sum()),
             "n_dropped_target_or_dup": int(n_raw - len(df))}
    return df, stats


def main():
    cfg = get_config("Stage 1: load & clean QMOF csv")
    cfg.ensure_dirs()
    log = setup_logging(cfg, "data_loading")
    df, stats = load_and_clean(cfg)
    df.drop(columns=["comp_dict"]).to_csv(cfg.processed_csv, index=False)
    log.info("saved %s | %s", cfg.processed_csv, stats)


if __name__ == "__main__":
    main()
