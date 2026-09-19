"""Stage 2: EDA on the cleaned table.
Outputs: figures/eda_*.png, metrics/eda_stats.json
Run:     python -m src.eda
"""
from __future__ import annotations
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import get_config
from .utils import comp_dict, setup_logging


def main():
    cfg = get_config("Stage 2: EDA")
    cfg.ensure_dirs()
    log = setup_logging(cfg, "eda")
    df = pd.read_csv(cfg.processed_csv)
    df["comp_dict"] = df["formula"].map(comp_dict)

    stats = df[["bandgap", "pld", "lcd", "density", "energy_per_atom",
                "natoms_formula"]].describe().to_dict()
    stats["metal_fraction_lt_thr"] = float((df["bandgap"] < cfg.metal_thr).mean())
    stats["metal_threshold_ev"] = cfg.metal_thr
    log.info("metal fraction (gap < %.2f eV): %.3f", cfg.metal_thr,
             stats["metal_fraction_lt_thr"])

    # distributions
    fig, ax = plt.subplots(1, 4, figsize=(20, 4))
    ax[0].hist(df["bandgap"], bins=80, color="steelblue"); ax[0].set_title("PBE band gap (eV)")
    ax[1].hist(df["pld"], bins=60, color="seagreen");      ax[1].set_title("PLD (Å)")
    ax[2].hist(df["lcd"], bins=60, color="seagreen");      ax[2].set_title("LCD (Å)")
    ax[3].hist(df["density"], bins=60, color="indianred"); ax[3].set_title("Density (g/cm³)")
    plt.tight_layout()
    p = os.path.join(cfg.figures_dir, "eda_distributions.png")
    plt.savefig(p, dpi=150); plt.close(); log.info("saved %s", p)

    # correlations + LCD vs gap
    feat = ["bandgap", "pld", "lcd", "density", "energy_per_atom", "natoms_formula"]
    corr = df[feat].corr()
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    im = ax[0].imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax[0].set_xticks(range(len(feat))); ax[0].set_xticklabels(feat, rotation=45, ha="right")
    ax[0].set_yticks(range(len(feat))); ax[0].set_yticklabels(feat)
    for i in range(len(feat)):
        for j in range(len(feat)):
            ax[0].text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax[0]); ax[0].set_title("Pearson correlation")
    ax[1].hexbin(df["lcd"], df["bandgap"], gridsize=50, cmap="viridis", mincnt=1)
    ax[1].set_xlabel("LCD (Å)"); ax[1].set_ylabel("Band gap (eV)")
    plt.tight_layout()
    p = os.path.join(cfg.figures_dir, "eda_corr.png")
    plt.savefig(p, dpi=150); plt.close(); log.info("saved %s", p)

    # mean gap for MOFs containing the 15 most common elements
    el_gaps = defaultdict(list)
    for cd, g in zip(df["comp_dict"], df["bandgap"]):
        if isinstance(cd, dict):
            for s in cd:
                el_gaps[s].append(g)
    top = sorted(el_gaps, key=lambda s: -len(el_gaps[s]))[:15]
    means = [float(np.mean(el_gaps[s])) for s in top]
    order = np.argsort(means)
    plt.figure(figsize=(7, 5))
    plt.barh(np.array(top)[order], np.array(means)[order], color="teal")
    plt.xlabel("mean PBE gap (eV)"); plt.title("Band gap by element presence")
    plt.tight_layout()
    p = os.path.join(cfg.figures_dir, "eda_elements.png")
    plt.savefig(p, dpi=150); plt.close(); log.info("saved %s", p)

    with open(os.path.join(cfg.metrics_dir, "eda_stats.json"), "w") as f:
        json.dump(stats, f, indent=2, default=str)


if __name__ == "__main__":
    main()
