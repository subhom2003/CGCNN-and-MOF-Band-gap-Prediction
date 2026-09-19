"""Stage 5: train the main CGCNN.
Outputs: models/best_model.pt, metrics/history_cgcnn_main.csv,
         metrics/best_val.json, figures/learning_curves.png
Run:     python -m src.train [--epochs 300 --hidden 128 ...]
"""
from __future__ import annotations
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import get_config
from .engine import load_artifacts, make_loaders, train_model, save_checkpoint
from .utils import setup_logging, save_json


def main():
    cfg = get_config("Stage 5: train CGCNN")
    cfg.ensure_dirs()
    log = setup_logging(cfg, "train")

    graphs, splits, aux_cols, use_aux, aux_by_id, y_by_id = load_artifacts(cfg)
    if cfg.smoke_test:
        splits = {k: v[: max(10, len(v) // 20)] for k, v in splits.items()}
        log.warning("SMOKE TEST sizes: %s", {k: len(v) for k, v in splits.items()})

    loaders = make_loaders(cfg, graphs, splits, aux_by_id, y_by_id)
    model, hist, best_val = train_model(cfg, loaders, len(aux_cols),
                                        tag="cgcnn_main", logger=log)

    sc = np.load(os.path.join(cfg.splits_dir, "scaler.npz"))
    with open(os.path.join(cfg.splits_dir, "aux_medians.json")) as f:
        medians = json.load(f)
    model_path = os.path.join(cfg.model_dir, "best_model.pt")
    save_checkpoint(cfg, model, aux_cols, sc["mean"], sc["scale"], medians,
                    best_val, model_path, extra={"use_aux": use_aux})

    hist.to_csv(os.path.join(cfg.metrics_dir, "history_cgcnn_main.csv"), index=False)
    save_json({"best_val_mae": best_val, "n_aux": len(aux_cols)},
              os.path.join(cfg.metrics_dir, "best_val.json"))

    plt.figure(figsize=(7, 4))
    plt.plot(hist.epoch, hist.train_mae, label="train")
    plt.plot(hist.epoch, hist.val_mae, label="val")
    plt.xlabel("epoch"); plt.ylabel("MAE (eV)"); plt.legend()
    plt.title("Learning curves")
    plt.tight_layout()
    p = os.path.join(cfg.figures_dir, "learning_curves.png")
    plt.savefig(p, dpi=150); plt.close()

    log.info("saved %s | best val MAE %.4f eV", model_path, best_val)


if __name__ == "__main__":
    main()
