"""Shared engine: artifact loading, PyG loader assembly, train/eval loops,
checkpointing. Used by train.py, baselines.py, interpret.py, evaluate.py."""
from __future__ import annotations
import json
import os
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_absolute_error

from .config import DEVICE, set_seed
from .model import CGCNN


def load_artifacts(cfg):
    """graphs + splits + aux + table -> everything train/eval stages need."""
    graphs = torch.load(cfg.graph_cache, weights_only=False)
    with open(os.path.join(cfg.splits_dir, "splits.json")) as f:
        splits = json.load(f)
    with open(os.path.join(cfg.splits_dir, "aux_cols.json")) as f:
        aux_info = json.load(f)
    aux_cols, use_aux = aux_info["aux_cols"], bool(aux_info.get("use_aux", True))
    sc = np.load(os.path.join(cfg.splits_dir, "scaler.npz"))
    table = pd.read_csv(os.path.join(cfg.splits_dir, "table.csv"))

    X = table[aux_cols].values.astype(np.float32)
    X[~np.isfinite(X)] = np.nan
    if np.isnan(X).any():                             # safety net (stage 4 already imputes)
        colmed = np.nanmedian(X, axis=0)
        colmed = np.where(np.isfinite(colmed), colmed, 0.0)
        nan_i = np.where(np.isnan(X))
        X[nan_i] = colmed[nan_i[1]]
    Xs = ((X - sc["mean"]) / sc["scale"]).astype(np.float32)

    aux_by_id = {mid: Xs[i] for i, mid in enumerate(table["id"])}
    y_by_id = {mid: float(v) for mid, v in zip(table["id"], table["bandgap"])}
    return graphs, splits, aux_cols, use_aux, aux_by_id, y_by_id


def make_loaders(cfg, graphs, splits, aux_by_id, y_by_id):
    """Attach aux/y to graphs and wrap into DataLoaders (in-memory, fast)."""
    loaders = {}
    for name, ids in splits.items():
        ds = []
        for mid in ids:
            g = graphs[mid]
            g.aux = torch.tensor(aux_by_id[mid], dtype=torch.float).unsqueeze(0)
            g.y = torch.tensor([y_by_id[mid]], dtype=torch.float)
            g.mid = mid
            ds.append(g)
        loaders[name] = DataLoader(ds, batch_size=cfg.batch_size,
                                   shuffle=(name == "train"),
                                   num_workers=cfg.num_workers)
    return loaders


@torch.no_grad()
def predict_loader(model, loader, device=DEVICE):
    model.eval()
    P, Y, MIDS = [], [], []
    for batch in loader:
        batch = batch.to(device)
        P.append(model(batch).view(-1).cpu())
        Y.append(batch.y.view(-1).cpu())
        MIDS += list(batch.mid)
    return torch.cat(Y).numpy(), torch.cat(P).numpy(), MIDS


def train_one_epoch(model, loader, opt, cfg, device=DEVICE):
    model.train()
    tot = n = 0
    for batch in loader:
        batch = batch.to(device)
        loss = F.l1_loss(model(batch).view(-1), batch.y.view(-1))   # L1 = MAE loss
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        tot += float(loss) * batch.num_graphs
        n += batch.num_graphs
    return tot / n


def train_model(cfg, loaders, aux_dim, tag="cgcnn", verbose=True, logger=None):
    """Train with L1 loss + early stopping on val MAE. Returns (model, history, best_mae)."""
    set_seed(cfg.seed)
    device = DEVICE
    model = CGCNN(cfg, aux_dim=aux_dim).to(device)
    if cfg.optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum,
                              weight_decay=cfg.weight_decay)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                weight_decay=cfg.weight_decay)
    sched = ExponentialLR(opt, gamma=cfg.lr_gamma)

    best_mae, best_state, best_ep, hist = float("inf"), None, -1, []
    for ep in range(1, cfg.epochs + 1):
        tr = train_one_epoch(model, loaders["train"], opt, cfg, device)
        yv, pv, _ = predict_loader(model, loaders["val"], device)
        va = float(mean_absolute_error(yv, pv))
        sched.step()
        hist.append({"epoch": ep, "train_mae": tr, "val_mae": va,
                     "lr": sched.get_last_lr()[0]})
        if va < best_mae - 1e-4:
            best_mae, best_ep = va, ep
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        if verbose and (ep % 10 == 0 or ep == 1):
            msg = (f"[{tag}] ep {ep:3d} | train {tr:.3f} | val {va:.3f} "
                   f"| best {best_mae:.3f}")
            (logger.info if logger else print)(msg)
        if ep - best_ep >= cfg.patience:
            msg = (f"[{tag}] early stop ep {ep} (best ep {best_ep}, "
                   f"val MAE {best_mae:.3f})")
            (logger.info if logger else print)(msg)
            break

    model.load_state_dict(best_state)
    return model, pd.DataFrame(hist), float(best_mae)


def save_checkpoint(cfg, model, aux_cols, scaler_mean, scaler_scale, medians,
                    best_val, path, extra=None):
    d = asdict(cfg)
    d["fc_hidden"] = list(d["fc_hidden"])
    ckpt = {"model_state": model.state_dict(), "cfg": d, "aux_cols": aux_cols,
            "scaler_mean": np.asarray(scaler_mean),
            "scaler_scale": np.asarray(scaler_scale),
            "medians": medians, "best_val_mae": best_val}
    if extra:
        ckpt.update(extra)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(ckpt, path)
    return path
