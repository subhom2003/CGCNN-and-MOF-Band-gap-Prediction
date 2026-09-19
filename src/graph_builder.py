"""Stage 3: CIF -> PyG crystal graphs (CGCNN representation), parallel + cached.

Nodes = atoms (atomic numbers); edges = all PBC neighbors within `cutoff`
(<= max_neighbors per atom, both directions stored, distance per edge).
Outputs: outputs/graphs/graphs.pt (dict id->Data), natoms_map.json,
         graphs.pt.meta.json, metrics/graph_failures.csv
Run:     python -m src.graph_builder
"""
from __future__ import annotations
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from pymatgen.core import Structure
from tqdm import tqdm

from .config import get_config
from .utils import setup_logging


def build_graph(cif_path: str, cutoff: float, max_neighbors: int, max_natoms: int):
    """CIF file -> torch_geometric Data(z, edge_index, edge_dist, natoms)."""
    try:
        struct = Structure.from_file(cif_path)
    except Exception as e:
        return None, None, f"PARSE_FAIL: {type(e).__name__}: {e}"
    n = len(struct)
    if n == 0:
        return None, None, "EMPTY"
    if n > max_natoms:
        return None, None, f"TOO_BIG({n})"

    z = torch.tensor([int(s.specie.Z) for s in struct], dtype=torch.long)

    # flat neighbor list under PBC (includes periodic self-images: correct for small cells)
    ci, ni, _, dist = struct.get_neighbor_list(r=cutoff)
    m = dist >= 0.1                                   # kill degenerate overlaps
    ci, ni, dist = ci[m], ni[m], dist[m]

    # cap neighbors per atom: sort by (center, distance), keep first k per center
    order = np.lexsort((dist, ci))
    ci, ni, dist = ci[order], ni[order], dist[order]
    if len(ci):
        starts = np.r_[True, ci[1:] != ci[:-1]]
        gstart = np.where(starts)[0]
        within = np.arange(len(ci)) - gstart[np.cumsum(starts) - 1]
        keep = within < max_neighbors
        ci, ni, dist = ci[keep], ni[keep], dist[keep]

    # undirected multigraph: store both directions of every pair
    src = np.concatenate([ni, ci])
    dst = np.concatenate([ci, ni])
    d2 = np.concatenate([dist, dist]).astype(np.float32)

    data = Data(z=z,
                edge_index=torch.tensor(np.vstack([src, dst]), dtype=torch.long),
                edge_dist=torch.tensor(d2))
    data.natoms = int(n)
    return data, n, None


def _job(args):
    mid, path, cutoff, kmax, nmax = args
    data, n, err = build_graph(path, cutoff, kmax, nmax)
    return mid, data, n, err


def main():
    cfg = get_config("Stage 3: build crystal graphs")
    cfg.ensure_dirs()
    log = setup_logging(cfg, "graph_builder")
    df = pd.read_csv(cfg.processed_csv)
    meta = {"cutoff": cfg.cutoff, "max_neighbors": cfg.max_neighbors,
            "max_natoms": cfg.max_natoms}
    meta_path = cfg.graph_cache + ".meta.json"

    if os.path.exists(cfg.graph_cache):
        graphs = torch.load(cfg.graph_cache, weights_only=False)
        log.info("loaded cache with %d graphs", len(graphs))
        if os.path.exists(meta_path):
            old = json.load(open(meta_path))
            if old != meta:
                log.warning("cache built with different params %s (now %s). "
                            "Delete %s to rebuild if unintended.", old, meta, cfg.graph_cache)
    else:
        tasks = [(r["id"], r["cif_path"], cfg.cutoff, cfg.max_neighbors, cfg.max_natoms)
                 for _, r in df[df["has_cif"]].iterrows()]
        graphs, failures = {}, []
        t0 = time.time()
        with Pool(cfg.n_jobs) as pool:
            it = pool.imap_unordered(_job, tasks, chunksize=32)
            for mid, data, n, err in tqdm(it, total=len(tasks), desc="CIF->graph"):
                if data is None:
                    failures.append({"id": mid, "error": err})
                else:
                    data.mid = mid
                    graphs[mid] = data
        log.info("built %d graphs in %.0fs | failures: %d",
                 len(graphs), time.time() - t0, len(failures))
        torch.save(graphs, cfg.graph_cache)
        json.dump(meta, open(meta_path, "w"))
        pd.DataFrame(failures, columns=["id", "error"]).to_csv(
            os.path.join(cfg.metrics_dir, "graph_failures.csv"), index=False)
        log.info("saved %s", cfg.graph_cache)

    with open(cfg.natoms_map, "w") as f:
        json.dump({mid: int(g.natoms) for mid, g in graphs.items()}, f)
    log.info("rows with graphs: %d / %d", int(df["id"].isin(graphs).sum()), len(df))


if __name__ == "__main__":
    main()
