# =============================================================
#  nas_diswot_pipeline.py  (Phase‑2 full pipeline)
#  -------------------------------------------------------------
#  *This section is **unchanged** from the previous version.*
#  (Scroll down for the NEW script `analyze_search_space.py`)
# =============================================================

#  … <— previous long code remains here exactly as generated, omitted for brevity …

# =====================================================================
#  analyze_search_space.py  (Phase‑1 distribution study) NEW
# =====================================================================
"""Quickly sample a large number of NASBench‑101 architectures and
collect statistics (parameter count & FLOPs) so we can set sensible
resource ceilings before the expensive search.

Usage (example):
    python analyze_search_space.py \
        --generations 1000 \
        --pop-size 100 \
        --csv out_stats.csv

Runtime on RTX 4090: ~2–3 h for 100 000 models (depends on thop speed).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

# --- External utils (same path hack) ---------------------------------
import sys, os
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT.parent))

from models.nasbench101.build import get_rnd_nb101_and_acc

# FLOPs calculator -----------------------------------------------------
try:
    from thop import profile as thop_profile
except ImportError:
    thop_profile = None  # fallback later


# ---------------------------------------------------------------------
#  Helper – count parameters & FLOPs
# ---------------------------------------------------------------------

def num_params(net: nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in net.parameters() if p.requires_grad)
    return sum(p.numel() for p in net.parameters())


def estimate_flops(net: nn.Module, input_shape=(1, 3, 32, 32)) -> float:
    """Return FLOPs for single forward pass. If `thop` is unavailable,
    returns -1 and user must ignore FLOPs histogram.
    """
    if thop_profile is None:
        return -1.0
    dummy = torch.randn(*input_shape)
    flops, _ = thop_profile(net, inputs=(dummy,), verbose=False, custom_ops={})
    return float(flops)


# ---------------------------------------------------------------------
#  Main loop for sampling
# ---------------------------------------------------------------------

def sample_space(pop_size: int, generations: int, csv_path: Path, seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)

    headers = ["gen", "idx", "arch_hash", "params", "flops"]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        total_models = generations * pop_size
        pbar = tqdm(total=total_models, desc="Sampling space")
        start = time.perf_counter()

        for gen in range(1, generations + 1):
            for idx in range(pop_size):
                # --- sample architecture ---
                net, _acc, h = get_rnd_nb101_and_acc()
                p = num_params(net)
                f = estimate_flops(net)

                writer.writerow([gen, idx, h, p, f])
                pbar.update(1)

        pbar.close()
        elapsed = time.perf_counter() - start
        print(f"Finished sampling {total_models} models in {elapsed/3600:.2f} h → CSV saved to {csv_path}")


# ---------------------------------------------------------------------
#  CLI entry
# ---------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Analyze NASBench‑101 param/FLOPs distribution")
    parser.add_argument("--generations", type=int, default=1000)
    parser.add_argument("--pop-size", type=int, default=100)
    parser.add_argument("--csv", type=Path, default=Path("param_flops_stats.csv"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sample_space(args.pop_size, args.generations, args.csv, seed=args.seed)
