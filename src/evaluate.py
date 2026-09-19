"""Stage 6: evaluate trained model on the test split.
Outputs: metrics/test_metrics.json, figures/test_analysis.png,
         metrics/classification_report.txt, metrics/worst_predictions.csv
Run:     python -m src.evaluate [--model_path outputs/models/best_model.pt]
"""
from __future__ import annotations
import os

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, accuracy_score,
                             classification_report, mean_absolute_error)

from .config import get_config, CFG, DEVICE
from .engine import load_artifacts, make_loaders, predict_loader
from .model import CGCNN
from .utils import reg_metrics, setup_logging, save_json


def load_model_from_ckpt(ckpt, device=DEVICE):
    c = CFG(**ckpt["cfg"])
    model = CGCNN(c, aux_dim=len(ckpt["aux_cols"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, c


def main():
    cfg = get_config("Stage 6: evaluate on test split")
    cfg.ensure_dirs()
    log = setup_logging(cfg, "evaluate")

    model_path = cfg.model_path or os.path.join(cfg.model_dir, "best_model.pt")
    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model, mcfg = load_model_from_ckpt(ckpt)
    log.info("loaded %s (best val MAE %.4f)", model_path, ckpt["best_val_mae"])

    graphs, splits, aux_cols, use_aux, aux_by_id, y_by_id = load_artifacts(cfg)
    loaders = make_loaders(mcfg, graphs, {"test": splits["test"]}, aux_by_id, y_by_id)
    yte, pte, mids = predict_loader(model, loaders["test"])

    m = reg_metrics(yte, pte)
    log.info("TEST: %s", {k: round(v, 4) for k, v in m.items()})
    save_json(m, os.path.join(cfg.metrics_dir, "test_metrics.json"))

    # parity / residuals / error-vs-gap
    fig, ax = plt.subplots(1, 3, figsize=(19, 5))
    ax[0].hexbin(yte, pte, gridsize=60, cmap="viridis", mincnt=1)
    lim = [0, max(yte.max(), pte.max()) * 1.05]
    ax[0].plot(lim, lim, "r--"); ax[0].set_xlim(lim); ax[0].set_ylim(lim)
    ax[0].set_xlabel("DFT PBE gap (eV)"); ax[0].set_ylabel("Predicted (eV)")
    ax[0].set_title(f"Parity (MAE={m['MAE']:.3f} eV, R²={m['R2']:.3f})")
    res = yte - pte
    ax[1].hist(res, bins=60, color="slateblue"); ax[1].axvline(0, color="r", ls="--")
    ax[1].set_xlabel("residual DFT − ML (eV)"); ax[1].set_title("Residuals")
    bins_e = np.arange(0, np.ceil(yte.max()) + 1, 1.0)
    ids_b = np.digitize(yte, bins_e)
    mae_b = [mean_absolute_error(yte[ids_b == i], pte[ids_b == i])
             if (ids_b == i).sum() > 5 else np.nan
             for i in range(1, len(bins_e))]
    ax[2].bar(range(1, len(bins_e)), mae_b, color="darkorange")
    ax[2].set_xlabel("DFT gap bin (eV)"); ax[2].set_ylabel("MAE (eV)")
    plt.tight_layout()
    p = os.path.join(cfg.figures_dir, "test_analysis.png")
    plt.savefig(p, dpi=150); plt.close(); log.info("saved %s", p)

    # metal vs non-metal discrimination (paper Fig. 2d analog)
    cls_t = (yte >= cfg.metal_thr).astype(int)
    cls_p = (pte >= cfg.metal_thr).astype(int)
    auc = roc_auc_score(cls_t, pte)
    rep = classification_report(cls_t, cls_p, target_names=["metal", "non-metal"],
                                digits=3)
    with open(os.path.join(cfg.metrics_dir, "classification_report.txt"), "w") as f:
        f.write(f"ROC-AUC: {auc:.4f}\nAccuracy: {accuracy_score(cls_t, cls_p):.4f}\n\n")
        f.write(rep)
    log.info("ROC-AUC %.4f", auc)

    # worst 20 predictions
    table = pd.read_csv(os.path.join(cfg.splits_dir, "table.csv")).set_index("id")
    worst = np.argsort(-np.abs(res))[:20]
    wdf = pd.DataFrame({
        "id": [mids[i] for i in worst],
        "formula": [table.loc[mids[i], "formula"] for i in worst],
        "DFT": yte[worst], "ML": pte[worst], "abs_err": np.abs(res[worst])})
    wdf.to_csv(os.path.join(cfg.metrics_dir, "worst_predictions.csv"), index=False)
    log.info("saved worst_predictions.csv")


if __name__ == "__main__":
    main()
