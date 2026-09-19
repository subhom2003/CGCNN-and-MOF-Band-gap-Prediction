"""Stage 9: inference -> predictions.csv with columns: ID, Predicted_Value.

Usage:
  python -m src.predict                                   # test-split ids
  python -m src.predict --ids_file data/raw/mapping.csv   # competition ID list
  python -m src.predict --ids qmof-000dce3,qmof-019ba28   # explicit ids
"""
from __future__ import annotations
import json
import os

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from .config import get_config, CFG, DEVICE
from .graph_builder import build_graph
from .model import CGCNN
from .utils import COLS, comp_dict, setup_logging


def main():
    cfg = get_config("Stage 9: predict")
    cfg.ensure_dirs()
    log = setup_logging(cfg, "predict")

    model_path = cfg.model_path or os.path.join(cfg.model_dir, "best_model.pt")
    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
    mc = CFG(**ckpt["cfg"])
    model = CGCNN(mc, aux_dim=len(ckpt["aux_cols"])).to(DEVICE)
    model.load_state_dict(ckpt["model_state"]); model.eval()

    # resolve ids
    if cfg.ids:
        ids = [s.strip() for s in cfg.ids.split(",") if s.strip()]
    elif cfg.ids_file:
        mf = pd.read_csv(cfg.ids_file)
        col = "ID" if "ID" in mf.columns else mf.columns[0]
        ids = mf[col].astype(str).str.strip().tolist()
    else:
        with open(os.path.join(cfg.splits_dir, "splits.json")) as f:
            ids = json.load(f)["test"]
    log.info("predicting %d structures", len(ids))

    # 1) graphs (build any missing)
    graphs = torch.load(cfg.graph_cache, weights_only=False)
    todo = [i for i in ids if i not in graphs]
    if todo:
        log.info("building %d missing graphs", len(todo))
        for mid in tqdm(todo):
            p = os.path.join(cfg.cif_dir, f"{mid}.cif")
            g, n, err = build_graph(p, mc.cutoff, mc.max_neighbors, mc.max_natoms)
            if g is None:
                log.warning("SKIP %s (%s)", mid, err)
                continue
            g.mid = mid
            graphs[mid] = g
    keep = [i for i in ids if i in graphs]

    # 2) aux features: join from raw csv when available, else train medians
    med = ckpt["medians"]
    aux_cols = ckpt["aux_cols"]
    src = None
    if os.path.exists(cfg.raw_csv):
        raw = pd.read_csv(cfg.raw_csv)
        raw = raw[[c for c in COLS if c in raw.columns]].rename(columns=COLS)
        raw["id"] = raw["id"].astype(str).str.strip()
        src = raw.set_index("id")

    natoms = {mid: int(g.natoms) for mid, g in graphs.items()}
    rows, missing_aux = [], 0
    for mid in keep:
        vec = dict(med)
        if src is not None and mid in src.index:
            r = src.loc[mid]
            cd = comp_dict(r["formula"])
            nat = natoms.get(mid) or (sum(cd.values()) if cd else 1)
            vec = {
                "pld": 0.0 if pd.isna(r["pld"]) else float(r["pld"]),
                "lcd": 0.0 if pd.isna(r["lcd"]) else float(r["lcd"]),
                "density": (float(r["density"]) if pd.notna(r["density"])
                            else med.get("density", 0.0)),
                "pld_missing": float(pd.isna(r["pld"])),
                "lcd_missing": float(pd.isna(r["lcd"])),
                "energy_per_atom": float(r["energy_total"]) / nat,
            }
        else:
            missing_aux += 1
        rows.append([float(vec.get(c, med.get(c, 0.0))) for c in aux_cols])
    A = ((np.array(rows, np.float32) - ckpt["scaler_mean"])
         / ckpt["scaler_scale"]).astype(np.float32)

    # 3) predict
    ds = []
    for mid in keep:
        g = graphs[mid]
        g.mid = mid
        ds.append(g)
    preds, k = {}, 0
    with torch.no_grad():
        for batch in DataLoader(ds, batch_size=mc.batch_size):
            batch = batch.to(DEVICE)
            b = batch.num_graphs
            batch.aux = torch.tensor(A[k:k + b], device=DEVICE)
            k += b
            preds.update(zip(batch.mid, model(batch).view(-1).cpu().tolist()))

    out = pd.DataFrame({"ID": ids,
                        "Predicted_Value": [preds.get(i, np.nan) for i in ids]})
    n_missing = int(out["Predicted_Value"].isna().sum())
    if n_missing:
        log.warning("%d ids had no graph — filling with mean prediction", n_missing)
        out["Predicted_Value"] = out["Predicted_Value"].fillna(
            float(np.nanmean(list(preds.values()))))
    out_path = os.path.join(cfg.predictions_dir, "predictions.csv")
    out.to_csv(out_path, index=False)
    log.info("wrote %s | missing graphs: %d | ids w/o csv aux (medians used): %d",
             out_path, n_missing, missing_aux)


if __name__ == "__main__":
    main()
