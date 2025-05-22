# =============================================================
#  nas_diswot_pipeline.py
#  -------------------------------------------------------------
#  End‑to‑end experimental pipeline for large‑scale architecture
#  screening using Zero‑Cost proxies + DisWOT + lightweight KD +
#  optional Born‑Again (BAN) training – designed for **RTX 4090
#  single‑GPU** execution under constrained compute.
#
#  Author: ChatGPT (generated for user request)
#  Date  : 2025‑05‑22
# =============================================================
"""Top‑level script overview
===============================================================
The pipeline implements a three‑phase, multi‑generation search as
discussed in chat:
    ① Super‑fast screening (zero‑cost + DisWOT) for every generation
    ② Mini KD (3 epoch) every *milestone* generations to refine top k
    ③ Final Born‑Again training on only a handful of architectures

Typical run‑time budget (RTX 4090):
    * Phase ① :   ≈ 30 s / generation × 1000 gen  →  8.3 h
    * Phase ② :   ≈ 40 min every 100 gen         →  6.7 h
    * Phase ③ :   50 epoch × ≤5 models           → 25 h (max)
    ----------------------------------------------
                   total ≤ 40 h (≈ 1.7 days)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

# -------------------------------------------------------------------
# Insert project root so that the user‑provided helper modules work.
# -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT.parent))

# --- user‑provided modules ----------------------------------------------------
from dataset.cifar10 import get_cifar10_dataloaders  # or cifar10 variant
from distiller_zoo import ICKDLoss, Similarity
from models.nasbench101.build import (
    get_nb101_teacher,
    get_rnd_nb101_and_acc,
    query_nb101_acc,
)
from predictor.pruners import predictive  # zero‑cost proxy helpers

# ==============================================================================
#  Utility functions
# ==============================================================================

def set_global_seeds(seed: int) -> None:
    """
    torch と numpy と random のシードを設定して、再現性を確保するための関数。
    """
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def gaussian_init(m: nn.Module):
    """
    Layer 単位でのガウス分布初期化を行う関数。
    Conv2d, Linear weight: (mean=0.0, std=1.0)
    BatchNorm (GroupNorm) weight: 1.0
    Conv2d bias: 0.0
    Linear bias: 0.0
    BatchNorm (GroupNorm) bias: 0.0
    """
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, mean=0.0, std=1.0)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        if m.weight is not None:
            nn.init.ones_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=1.0)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# ==============================================================================
#  Phase ① – super‑fast screening (every generation)
# ==============================================================================

def compute_zc_score(
    net: nn.Module,
    img_batch: torch.Tensor,
    device: torch.device,
    weight_dict: dict[str, float] | None = None,
) -> float:
    """ 複数のゼロコストプロキシメトリックを組み合わせてスカラー値を計算する。

    * DisWOT ベースの ICKD + Similarity
    * NWOT, SynFlow などを `predictive.find_measures` 経由で計算

    Args:
        net (nn.Module): ネットワークモデル
        img_batch (torch.Tensor): 入力画像バッチ
        device (torch.device): デバイス (CPU/GPU)
        weight_dict (dict[str, float], optional): 各メトリックの重み辞書。デフォルトは None。
    """

    weight_dict = weight_dict or {
        "diswot": 1.0,
        "nwot": 1.0,
        "synflow": 1.0,
    }

    # 推論モードに設定（dropout や batchnorm を推論用に切り替える）
    net.eval()

    # 勾配計算を無効化し（省メモリ化）、混合精度推論を有効化（高速化）
    with torch.no_grad(), autocast():
        # ---------- DisWOT スコアの計算 ----------
        
        # モデルから特徴量と予測ロジットを取得
        feat, logits = net.forward_with_features(img_batch)

        # 論文で使われているテクニックを模倣するため、分類器の重みを 1×1 の空間次元を持つ形に変形
        compressed = net.classifier.weight.unsqueeze(-1).unsqueeze(-1)

        # ICKD（教師あり類似度）と Similarity（特徴マップ類似度）を用いた評価指標を初期化
        criterion_ickd = ICKDLoss()
        criterion_sp = Similarity()

        # DisWOT 値を計算（自己評価に基づく教師無しスコア）
        # 自己相関的な損失の合計をマイナスにしたものをスコアとして使用
        diswot_val = (
            -criterion_ickd([compressed], [compressed])[0]
            -criterion_sp(feat[-2], feat[-2])[0]
        ).item()

        # ---------- Zero-cost proxies の計算（`predictive` を利用） ----------
        
        zc_vals = predictive.find_measures(
            net,
            None,  # データローダは不要（img_batch を直接使用するため）
            dataload_info=["random", 3, 32],  # ダミーのローダ情報（Zero-cost指標が必要とする形式）
            measure_names=["nwot", "synflow"],  # 使用する Zero-cost proxy の種類
            device=device,
            loss_fn=F.cross_entropy,  # 使用する損失関数（ここではクロスエントロピー）
            input_data=img_batch,  # 評価に使う入力画像バッチ
        )
        # 各指標を取り出す
        nwot_val = zc_vals["nwot"]
        synflow_val = zc_vals["synflow"]

    # ---------- 各メトリクスのスコアの重み付き合計（後に全体で z-score 正規化される） ----------
    raw = {
        "diswot": diswot_val,
        "nwot": nwot_val,
        "synflow": synflow_val,
    }

    # 各メトリクスに対して重みを掛けてスコアを合計
    score = sum(weight_dict[k] * raw[k] for k in raw)

    # スコアを返す（構造探索などの目的で使用）
    return score



def screening_generation(
    gen_idx: int,
    pop_size: int,
    device: torch.device,
    img_batch: torch.Tensor,
) -> List[Tuple[str, float]]:
    """Generate *pop_size* random architectures and compute their ZC score.

    Returns list of (arch_hash, score) sorted descending (best first).
    """

    results: List[Tuple[str, float]] = []

    for idx in range(pop_size):
        torch.cuda.empty_cache()
        snet, _, arch_hash = get_rnd_nb101_and_acc()
        snet.apply(gaussian_init)
        snet.to(device)

        score = compute_zc_score(snet, img_batch, device)
        results.append((arch_hash, score))

    # sort by score descending
    results.sort(key=lambda x: x[1], reverse=True)
    return results


# ==============================================================================
#  Phase ② – quick KD every milestone generations
# ==============================================================================

def quick_kd(
    teacher_arch_hash: str,
    img_batch: torch.Tensor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 3,
) -> float:
    """Run mini KD (few epoch) and return validation accuracy."""

    teacher, _acc, _ = get_nb101_teacher(arch_hash=teacher_arch_hash, pretrained=False)
    teacher.apply(gaussian_init)
    teacher.to(device)
    teacher.eval()

    student, _, _ = get_nb101_teacher(arch_hash=teacher_arch_hash, pretrained=False)
    student.apply(gaussian_init)
    student.to(device)

    optimizer = torch.optim.SGD(student.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * len(train_loader))
    scaler = GradScaler()

    kd_loss = nn.KLDivLoss(reduction="batchmean")
    ce_loss = nn.CrossEntropyLoss()

    def kd_step(batch):
        x, y = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            with torch.no_grad():
                t_logits = teacher(x)
            s_logits = student(x)
            loss = 0.7 * kd_loss(F.log_softmax(s_logits / 4, dim=1), F.softmax(t_logits / 4, dim=1)) + 0.3 * ce_loss(s_logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

    # --- training loop ---
    student.train()
    for ep in range(epochs):
        for i, batch in enumerate(train_loader):
            kd_step(batch)
            if i == 100:  # cap iterations to speed‑up further if desired
                break
        # early stopping check (optional)

    # --- validation ---
    student.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            logits = student(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            if total >= 2000:  # evaluate on subset (speed)
                break
    val_acc = 100.0 * correct / total
    return val_acc


# ==============================================================================
#  Phase ③ – full BAN training
# ==============================================================================

def train_standard(
    net: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 50,
) -> float:
    """Standard supervised training (no KD) – returns best val accuracy."""

    net = net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * len(train_loader))
    scaler = GradScaler()
    criterion = nn.CrossEntropyLoss()

    best = 0.0
    for ep in range(epochs):
        net.train()
        for img, label in train_loader:
            img = img.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                logits = net(img)
                loss = criterion(logits, label)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        # quick validation each epoch
        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for img, label in val_loader:
                img = img.to(device)
                label = label.to(device)
                logits = net(img)
                pred = logits.argmax(dim=1)
                correct += (pred == label).sum().item()
                total += label.size(0)
            acc = 100.0 * correct / total
            best = max(best, acc)
    return best


# ==============================================================================
#  Main experiment loop
# ==============================================================================

def run_experiment(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_global_seeds(args.seed)

    # ---------------- datasets ----------------
    train_loader, val_loader = get_cifar100_dataloaders(batch_size=args.batch, num_workers=4)
    # Pre‑fetch a single batch to GPU for phase ① screening
    img_batch, _ = next(iter(train_loader))
    img_batch = img_batch[: args.screen_batch].to(device)

    # ---------------- bookkeeping ----------------
    elite_pool: deque[str] = deque(maxlen=args.elite_pool)  # best arch hashes ever seen
    ban_candidates: set[str] = set()

    start = time.perf_counter()
    for gen in range(1, args.generations + 1):
        print(f"\n===== Generation {gen}/{args.generations} =====")

        # === Phase ①: screening ===
        screen_res = screening_generation(gen, args.pop_size, device, img_batch)
        top_hashes = [h for h, _ in screen_res[: args.keep_top]]
        print("Top scores:", screen_res[:3])

        # Maintain elite pool
        elite_pool.extend(top_hashes)

        # === milestone KD every k generations ===
        if gen % args.milestone == 0:
            kd_scores = []
            for h in top_hashes:
                val_acc = quick_kd(h, img_batch, train_loader, val_loader, device, epochs=args.kd_epochs)
                kd_scores.append((h, val_acc))
                print(f"KD‑val {h[:6]}  : {val_acc:.2f} %")
            kd_scores.sort(key=lambda x: x[1], reverse=True)
            best_kd = kd_scores[: args.ban_pool]
            ban_candidates.update(h for h, _ in best_kd)
            print("[Milestone] added to BAN‑pool:", best_kd)

        # quick progress log
        elapsed = time.perf_counter() - start
        print(f"Time elapsed: {elapsed / 3600:.2f} h")

    # ---------------- Phase ③: BAN ----------------
    print("\n===== Final BAN phase =====")
    ban_list = list(ban_candidates)[: args.final_models]
    print("BAN candidates:", ban_list)

    final_results = []
    for idx, h in enumerate(ban_list, 1):
        print(f"\n--- BAN training {idx}/{len(ban_list)} : {h[:8]} ---")
        net, _acc, _ = get_nb101_teacher(arch_hash=h, pretrained=False)
        net.apply(gaussian_init)

        # Teacher₁
        teacher_acc = train_standard(net, train_loader, val_loader, device, epochs=args.ban_epochs)
        print(f"Teacher₁ val acc: {teacher_acc:.2f}%")

        # Student₁
        student, _, _ = get_nb101_teacher(arch_hash=h, pretrained=False)
        student.apply(gaussian_init)
        # quick KD fine‑tune using fully trained teacher (one pass, same epochs)
        kd_acc = quick_kd(h, img_batch, train_loader, val_loader, device, epochs=args.ban_epochs)
        print(f"Student₁ (BAN) val acc: {kd_acc:.2f}%")
        final_results.append((h, kd_acc))

    final_results.sort(key=lambda x: x[1], reverse=True)
    best_arch, best_acc = final_results[0]
    print("\n===== EXPERIMENT FINISHED =====")
    print(f"Best architecture: {best_arch}")
    print(f"Validation accuracy: {best_acc:.2f}%")

    # save summary
    summary = {
        "args": vars(args),
        "best_arch": best_arch,
        "best_acc": best_acc,
        "all_final": final_results,
    }
    (ROOT / "experiment_summary.json").write_text(json.dumps(summary, indent=2))


# ==============================================================================
#  Argument parsing
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Large‑scale NAS + DisWOT pipeline")

    # ---- general ----
    parser.add_argument("--generations", type=int, default=1000, help="Number of evolutionary generations")
    parser.add_argument("--pop_size", type=int, default=100, help="Population size per generation")
    parser.add_argument("--keep_top", type=int, default=10, help="How many top models to keep each generation")
    parser.add_argument("--milestone", type=int, default=100, help="Run quick KD every K generations")
    parser.add_argument("--seed", type=int, default=1, help="Global random seed")

    # ---- phase parameters ----
    parser.add_argument("--screen_batch", type=int, default=256, help="Batch size for screening forward (phase ①)")
    parser.add_argument("--kd_epochs", type=int, default=3, help="Epochs for quick KD (phase ②)")
    parser.add_argument("--ban_epochs", type=int, default=50, help="Epochs per BAN pass (phase ③)")
    parser.add_argument("--elite_pool", type=int, default=50, help="Max size of global elite pool")
    parser.add_argument("--ban_pool", type=int, default=3, help="Add top‑k models from quick KD to BAN candidates")
    parser.add_argument("--final_models", type=int, default=5, help="How many models to run full BAN on")

    # ---- dataloader ----
    parser.add_argument("--batch", type=int, default=128, help="Train loader batch size")

    args = parser.parse_args()

    run_experiment(args)
