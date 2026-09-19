"""Stage 7: baselines (mean, tabular GBM on aux+composition) + ablations
(Eq4 vs Eq5, aux on/off — each retrains a full CGCNN).
Outputs: metrics/results_summary.csv, models/best_model_eq4_aux.pt,
         models/best_model_noaux_eq5.pt, models/best_model_noaux_eq4.pt
Run:     python -m src.baselines [--skip_ablations true]
"""
from __future__ import annotations
import json
import os
from dataclasses import replace

import numpy as np
import pandas as pd
import torch
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from pymatgen.core import Element

from .config import get_config, CFG, DEVICE
from .engine import (load_artifacts, make_loaders, predict_loader,
                     train_model, save_checkpoint)
from .model import CGCNN
from .utils import reg_metrics, setup_logging, comp_dict

Z_OF = {Element.from_Z(z).symbol: z - 1 for z in range(1, 119)}


def comp_vec(formula):
    v = np.zeros(118, np.float32)
    cd = comp_dict(formula)
    if cd:
        for s, a in cd.items():
            if s in Z_OF:
                v[Z_OF[s]] += a
    return v


def main():
    cfg = get_config("Stage 7: baselines & ablations")
    cfg.ensure_dirs()
    log = setup_logging(cfg, "baselines")

    graphs, splits, aux_cols, use_aux, aux_by_id, y_by_id = load_artifacts(cfg)
    table = pd.read_csv(os.path.join(cfg.splits_dir, "table.csv"))

    # tabular design matrix: aux (raw) + composition one-hot counts
    Xaux = table[aux_cols].values.astype(np.float32)
    Xaux[~np.isfinite(Xaux)] = 0.0
    Cvec = np.stack([comp_vec(f) for f in table["formula"]])
    Xtab = np.hstack([Xaux, Cvec]).astype(np.float32)

    ids = table["id"].tolist()
    pos = {mid: i for i, mid in enumerate(ids)}
    tr = np.array([pos[m] for m in splits["train"]])
    te = np.array([pos[m] for m in splits["test"]])
    y = table["bandgap"].values.astype(np.float32)

    results = {}
    dummy = DummyRegressor(strategy="mean").fit(Xtab[tr], y[tr])
    results["Mean predictor"] = reg_metrics(y[te], dummy.predict(Xtab[te]))
    log.info("Mean predictor: %s", results["Mean predictor"])

    hgb = HistGradientBoostingRegressor(loss="absolute_error", max_iter=500,
                                        learning_rate=0.06, random_state=cfg.seed)
    hgb.fit(Xtab[tr], y[tr])
    results["GBM (aux+composition)"] = reg_metrics(y[te], hgb.predict(Xtab[te]))
    log.info("GBM: %s", results["GBM (aux+composition)"])

    # main CGCNN test metrics (reuse evaluate output if present, else compute)
    mpath = os.path.join(cfg.metrics_dir, "test_metrics.json")
    if os.path.exists(mpath):
        with open(mpath) as f:
            results["CGCNN (Eq5 + aux)"] = json.load(f)
    else:
        ckpt = torch.load(os.path.join(cfg.model_dir, "best_model.pt"),
                          map_location="cpu", weights_only=False)
        mc = CFG(**ckpt["cfg"])
        model = CGCNN(mc, aux_dim=len(ckpt["aux_cols"])).to(DEVICE)
        model.load_state_dict(ckpt["model_state"])
        ld = make_loaders(mc, graphs, {"test": splits["test"]}, aux_by_id, y_by_id)
        yt, pt, _ = predict_loader(model, ld["test"])
        results["CGCNN (Eq5 + aux)"] = reg_metrics(yt, pt)
    log.info("CGCNN: %s", results["CGCNN (Eq5 + aux)"])

    if not cfg.skip_ablations:
        sc = np.load(os.path.join(cfg.splits_dir, "scaler.npz"))
        with open(os.path.join(cfg.splits_dir, "aux_medians.json")) as f:
            medians = json.load(f)
        loaders_full = make_loaders(cfg, graphs, splits, aux_by_id, y_by_id)
        ablations = {
            "CGCNN Eq4 + aux": (replace(cfg, conv_type="simple"),
                                "best_model_eq4_aux.pt"),
            "CGCNN Eq5 no-aux": (replace(cfg, use_aux=False),
                                 "best_model_noaux_eq5.pt"),
            "CGCNN Eq4 no-aux": (replace(cfg, conv_type="simple", use_aux=False),
                                 "best_model_noaux_eq4.pt"),
        }
        for tag, (c_ab, fname) in ablations.items():
            log.info("training ablation: %s", tag)
            model_ab, _, bv = train_model(c_ab, loaders_full, len(aux_cols),
                                          tag=tag.replace(" ", "_"), logger=log)
            ld = make_loaders(c_ab, graphs, {"test": splits["test"]},
                              aux_by_id, y_by_id)
            yt, pt, _ = predict_loader(model_ab, ld["test"])
            results[tag] = reg_metrics(yt, pt)
            save_checkpoint(c_ab, model_ab, aux_cols, sc["mean"], sc["scale"],
                            medians, bv, os.path.join(cfg.model_dir, fname))
            log.info("%s -> %s", tag, results[tag])

    summary = pd.DataFrame(results).T
    summary.index.name = "model"
    out = os.path.join(cfg.metrics_dir, "results_summary.csv")
    summary.to_csv(out)
    log.info("saved %s\n%s", out, summary.round(4))


if __name__ == "__main__":
    main()
