"""Shared helpers: logging, column mapping, metrics, JSON IO."""
from __future__ import annotations
import json
import logging
import os
import sys

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pymatgen.core import Composition

# Raw QMOF columns we use -> short names
# (to train on HSE06 gaps instead: change the bandgap source column here)
COLS = {
    "qmof_id": "id",
    "info.formula": "formula",
    "info.pld": "pld",                              # pore-limiting diameter (A)
    "info.lcd": "lcd",                              # largest cavity diameter (A)
    "info.density": "density",                      # g/cm^3
    "outputs.pbe.energy_total": "energy_total",     # eV, extensive!
    "outputs.pbe.bandgap": "bandgap",               # TARGET (eV)
}


def setup_logging(cfg, name: str) -> logging.Logger:
    os.makedirs(cfg.logs_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(os.path.join(cfg.logs_dir, f"{name}.log"))
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def comp_dict(formula):
    """'Zn4C104H76N12O20' -> {'Zn':4.0, 'C':104.0, ...} or None on failure."""
    try:
        return {s: float(a) for s, a in Composition(str(formula).strip()).as_dict().items()}
    except Exception:
        return None


def reg_metrics(y_true, y_pred) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
