"""Central configuration: CFG dataclass + generic CLI override parser.

Every field can be overridden from the command line, e.g.:
    python -m src.train --epochs 50 --hidden 64 --smoke_test true
Booleans accept true/false/1/0/yes/no. Tuples: --fc_hidden 128,64
"""
from __future__ import annotations
import argparse
import json
import random
from dataclasses import dataclass, asdict, fields
import os
import numpy as np
import torch


def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("1", "true", "yes", "y")


@dataclass
class CFG:
    # ---- data paths ----
    raw_csv: str = "data/raw/qmof.csv"
    cif_dir: str = "data/raw/cifs"
    processed_csv: str = "data/processed/qmof_clean.csv"
    graph_cache: str = "outputs/graphs/graphs.pt"
    natoms_map: str = "outputs/graphs/natoms_map.json"
    splits_dir: str = "outputs/splits"
    model_dir: str = "outputs/models"
    figures_dir: str = "outputs/figures"
    metrics_dir: str = "outputs/metrics"
    predictions_dir: str = "outputs/predictions"
    logs_dir: str = "logs"
    # ---- task ----
    id_col: str = "id"
    target: str = "bandgap"          # = outputs.pbe.bandgap (eV)
    metal_thr: float = 0.1           # eV, PBE gap below this ~ metallic
    # ---- graph construction ----
    cutoff: float = 6.0              # neighbor radius (A), MOF-safe
    max_neighbors: int = 30
    gauss_bins: int = 41             # Gaussian edge-feature resolution
    max_natoms: int = 800
    n_jobs: int = 8                  # parallel CIF parsing
    # ---- model ----
    hidden: int = 128
    n_conv: int = 3
    fc_hidden: tuple = (128, 64)
    dropout: float = 0.0
    conv_type: str = "modified"      # "modified" = Eq.(5) | "simple" = Eq.(4)
    use_aux: bool = True             # fuse pld/lcd/density/e-per-atom
    use_energy: bool = True          # include energy_total/atom (DFT-derived!)
    linear_pool: bool = False        # interpretable variant
    # ---- training ----
    optimizer: str = "adamW"           # "sgd" (paper) | "adamw"
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.00001
    batch_size: int = 64
    epochs: int = 300
    lr_gamma: float = 0.98
    grad_clip: float = 5.0
    patience: int = 40               # early stopping
    num_workers: int = 0
    seed: int = 42
    smoke_test: bool = False         # train on a small subset
    # ---- stage-specific options ----
    model_path: str = ""             # "" -> model_dir/best_model.pt
    skip_ablations: bool = False
    ids: str = ""                    # comma-separated ids for predict
    ids_file: str = ""               # csv with an ID column for predict

    def ensure_dirs(self):
        for d in [self.splits_dir, self.model_dir, self.figures_dir,
                  self.metrics_dir, self.predictions_dir, self.logs_dir,
                  os.path.dirname(self.processed_csv),
                  os.path.dirname(self.graph_cache)]:
            os.makedirs(d, exist_ok=True)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def _load_json_defaults(path: str = "configs/default.json") -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def get_config(description: str = "") -> CFG:
    """CFG from JSON defaults, overridden by CLI `--field value`."""
    defaults = _load_json_defaults()
    cfg = CFG(**{k: v for k, v in defaults.items() if hasattr(CFG, k)})
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--print_config", action="store_true")
    for f in fields(CFG):
        t = type(f.default)
        if t is bool:
            kw = {"type": str2bool}
        elif f.name == "fc_hidden":
            kw = {"type": lambda s: tuple(int(x) for x in s.split(","))}
        elif t in (int, float, str):
            kw = {"type": t}
        else:
            kw = {"type": str}
        p.add_argument(f"--{f.name}", **kw)
    args = p.parse_args()
    for f in fields(CFG):
        v = getattr(args, f.name)
        if v is not None:
            setattr(cfg, f.name, v)
    if args.print_config:
        print(json.dumps(asdict(cfg), indent=2, default=str))
    return cfg
