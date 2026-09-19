"""Stage 8: interpretability.
(a) linear-pooling CGCNN -> per-element site contributions (paper's Fig. 3 analog)
(b) permutation importance of auxiliary features on the validation set
Outputs: figures/site_contributions.png, metrics/site_contributions.json,
         metrics/aux_perm_importance.json, models/interp_model.pt
Run:     python -m src.interpret
"""
from __future__ import annotations
import json
import os
from collections import defaultdict
from dataclasses import replace

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from pymatgen.core import Element

from .config import get_config, CFG, DEVICE
from .engine import (load_artifacts, make_loaders, train_model,
                     save_checkpoint, predict_loader)
from .model import CGCNN
from .utils import setup_logging


@torch.no_grad()
def element_site_contributions(model, loader, device=DEVICE):
    model.eval()
    agg = defaultdict(list)
    for batch in loader:
        batch = batch.to(device)
        _, site = model(batch, return_sites=True)
        for z, s in zip(batch.z.cpu().tolist(), site.cpu().tolist()):
            agg[z].append(s)
    return ({Element.from_Z(z).symbol: float(np.mean(v)) for z, v in agg.items()},
            {Element.from_Z(z).symbol: int(len(v)) for z, v in agg.items()})


@torch.no_grad()
def aux_perm_importance(model, loader, cols, n_rep=3, device=DEVICE):
    yb, pb, _ = predict_loader(model, loader, device)
    base = float(mean_absolute_error(yb, pb))
    imp = {}
    for j, col in enumerate(cols):
        maes = []
        for _ in range(n_rep):
            P, Yl = [], []
            for batch in loader:
                batch = batch.to(device)
                a = batch.aux.clone()
                a[:, j] = a[torch.randperm(a.size(0), device=a.device), j]
                batch.aux = a
                P.append(model(batch).view(-1).cpu())
                Yl.append(batch.y.view(-1).cpu())
            maes.append(float(mean_absolute_error(torch.cat(Yl), torch.cat(P))))
        imp[col] = float(np.mean(maes) - base)
    return dict(sorted(imp.items(), key=lambda kv: -kv[1])), base


def main():
    cfg = get_config("Stage 8: interpretability")
    cfg.ensure_dirs()
    log = setup_logging(cfg, "interpret")
    graphs, splits, aux_cols, use_aux, aux_by_id, y_by_id = load_artifacts(cfg)

    # (a) interpretable linear-pooling model
    c_i = replace(cfg, linear_pool=True, use_aux=False)
    loaders_i = make_loaders(c_i, graphs, splits, aux_by_id, y_by_id)
    model_i, _, bv = train_model(c_i, loaders_i, len(aux_cols),
                                 tag="cgcnn_interp", logger=log)
    sc = np.load(os.path.join(cfg.splits_dir, "scaler.npz"))
    with open(os.path.join(cfg.splits_dir, "aux_medians.json")) as f:
        medians = json.load(f)
    save_checkpoint(c_i, model_i, aux_cols, sc["mean"], sc["scale"], medians,
                    bv, os.path.join(cfg.model_dir, "interp_model.pt"))

    test_loader = make_loaders(c_i, graphs, {"test": splits["test"]},
                               aux_by_id, y_by_id)["test"]
    site_mean, site_n = element_site_contributions(model_i, test_loader)
    top = sorted(site_mean, key=lambda s: -site_n[s])[:25]
    order = np.argsort([site_mean[s] for s in top])
    plt.figure(figsize=(7, 6))
    plt.barh(np.array(top)[order], np.array([site_mean[s] for s in top])[order],
             color="darkcyan")
    plt.xlabel("mean site contribution to band gap (eV)")
    plt.title("Which atoms push the gap up / down?")
    plt.tight_layout()
    p = os.path.join(cfg.figures_dir, "site_contributions.png")
    plt.savefig(p, dpi=150); plt.close()
    with open(os.path.join(cfg.metrics_dir, "site_contributions.json"), "w") as f:
        json.dump({"site_mean": site_mean, "site_counts": site_n,
                   "interp_val_mae": bv}, f, indent=2)
    log.info("saved site contributions | interp val MAE %.4f", bv)

    # (b) permutation importance of aux features (main model, val split)
    ckpt = torch.load(os.path.join(cfg.model_dir, "best_model.pt"),
                      map_location=DEVICE, weights_only=False)
    mc = CFG(**ckpt["cfg"])
    model = CGCNN(mc, aux_dim=len(ckpt["aux_cols"])).to(DEVICE)
    model.load_state_dict(ckpt["model_state"]); model.eval()
    val_loader = make_loaders(mc, graphs, {"val": splits["val"]},
                              aux_by_id, y_by_id)["val"]
    imp, base = aux_perm_importance(model, val_loader, aux_cols)
    with open(os.path.join(cfg.metrics_dir, "aux_perm_importance.json"), "w") as f:
        json.dump({"baseline_val_mae": base, "delta_mae": imp}, f, indent=2)
    log.info("baseline val MAE %.4f | ΔMAE: %s", base,
             {k: round(v, 4) for k, v in imp.items()})


if __name__ == "__main__":
    main()
