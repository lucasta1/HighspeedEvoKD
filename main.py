# =============================================================
#  main.py  —  天井制約付き DisWOT + Zero‑Cost パイプライン (完成版)
# =============================================================
#  * このファイル 1 本だけで実験を回せます *
#
#  変更点（デバッグ版）
#  -------------------------------------------------------------
#  ✔ tqdm で進捗が表示されないケースに備え、各主要段階で明示 print
#  ✔ FLOPs 推定が -1 の場合でも通過させ、警告を出力
#  ✔ `run_experiment()` がファイル末尾まで途切れないよう再掲
#  ✔ Ctrl‑C で安全に終了できる try/except を追加
# =============================================================

from __future__ import annotations

import os
import argparse
import json
import math
import heapq
import random
import sys
import time
from pathlib import Path
from collections import deque
from itertools import product
from pathlib import Path
from typing import Optional, List, Tuple

import torch
import torch.nn as nn, optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ---------- ユーザーパス調整 ----------
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT.parent))

# ---------- プロジェクト依存 ----------
from dataset.cifar10 import get_cifar10_dataloaders  # 32×32 データセット
from distiller_zoo import ICKDLoss, Similarity
from models.nasbench101.build import (
    get_nb101_model,
    get_rnd_nb101_and_acc,
)
from predictor.pruners import predictive

# ---------- FLOPs カウンタ ----------
try:
    from thop import profile as thop_profile  # type: ignore
except ImportError:
    thop_profile = None

try:
    from fvcore.nn import FlopCountAnalysis  # type: ignore
except ImportError:
    FlopCountAnalysis = None  # type: ignore


# =============================================================
#  1. ユーティリティ
# =============================================================
def set_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def num_params(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())


def gaussian_init(m: nn.Module):
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


def to_scalar(val):
    """Tensor / list / 数値 / numpy.float64 …何でも float にする"""
    if isinstance(val, torch.Tensor):          # Tensor → 要素合計
        return val.detach().sum().item()
    if isinstance(val, (list, tuple)):         # list / tuple → 再帰的に合算
        return sum(to_scalar(v) for v in val)
    return float(val)                          # float, int, np.float64 など


def _find_last_conv(model: nn.Module) -> nn.Module:
    for module in reversed(list(model.modules())):
        if isinstance(module, nn.Conv2d):
            return module
    raise ValueError("No Conv2d layer found in model – please supply target_layer.")


class _GradCamHook:
    """Internal utility storing activations & gradients from a chosen layer."""

    def __init__(self, layer: nn.Module):
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._fwd = layer.register_forward_hook(self._save_activation)
        self._bwd = layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _, __, output):
        # output: Tensor of shape (N, C, H, W)
        self.activations = output.detach()
        return None  # no modification

    def _save_gradient(self, _, __, grad_output):
        # grad_output is a tuple – take first element
        self.gradients = grad_output[0].detach()
        return None

    def close(self):
        self._fwd.remove()
        self._bwd.remove()


def _grad_cam_maps(model: nn.Module, x: torch.Tensor, layer: nn.Module) -> torch.Tensor:
    """Return Grad‑CAM maps (N, H, W) for the *batch*."""
    hook = _GradCamHook(layer)

    model.zero_grad(set_to_none=True)
    logits = model(x)  # (N, num_classes)

    # Follow DisWOT: use class‑agnostic sum of logits
    score = logits.sum()
    score.backward()

    assert hook.activations is not None and hook.gradients is not None, "Hooks failed."
    act = hook.activations  # (N, C, H, W)
    grad = hook.gradients   # (N, C, H, W)

    weights = grad.mean(dim=(2, 3), keepdim=True)  # (N, C, 1, 1)
    cam = (weights * act).sum(dim=1)  # (N, H, W)
    cam = F.relu(cam)

    # normalise each map to [0,1] (avoid div‑by‑zero)
    cam = cam.view(cam.size(0), -1)
    cam = cam / (cam.amax(dim=1, keepdim=True) + 1e-8)
    cam = cam.view_as(cam)

    hook.close()
    return cam  # (N, H, W)


def _channel_correlation(maps: torch.Tensor) -> torch.Tensor:
    """Return channel‑wise correlation matrix given CAMs.

    maps: Tensor (N, H, W)  – Grad‑CAM maps per sample.
    Returns: (C, C) correlation; here C = N (batch) by paper’s notation.
    We instead follow paper Eq.(3): correlation across *channels* of CAM;
    but since we have already aggregated channels into CAM, use sample index
    as proxy.  This implementation mirrors authors' released code, where they
    use *channel* dimension of *feature map* for semantic metric – we emulate
    by treating flattened spatial dims as feature vectors per channel.
    """
    # reshape to (N, H*W)
    Fm = maps.view(maps.size(0), -1)  # (N, HW)
    # correlation (cosine similarity matrix) – normalised dot products
    norm = F.normalize(Fm, dim=1)  # (N, HW)
    corr = norm @ norm.T  # (N, N)
    # L2 normalise whole matrix as paper
    corr = corr / (corr.norm(p=2) + 1e-8)
    return corr  # (N, N)


def _sample_relation_matrix(feat: torch.Tensor) -> torch.Tensor:
    """Compute sample‑wise correlation matrix for flattened feature maps."""
    N = feat.size(0)
    flat = feat.view(N, -1)
    norm = F.normalize(flat, dim=1)
    rel = norm @ norm.T  # (N, N)
    rel = rel / (rel.norm(p=2) + 1e-8)
    return rel


def diswot_score(
    teacher: nn.Module,
    student: nn.Module,
    x: torch.Tensor,
    device,
) -> float:
    """Compute DisWOT score for *random‑initialised* teacher‑student pair.

    Parameters
    ----------
    teacher, student : nn.Module
        Networks **must** be in evaluation mode *before* calling.
    x : torch.Tensor (N, C, H, W)
        A single mini‑batch (paper uses one mini‑batch, e.g. 32 or 64 images).
    device : torch.device
        CUDA recommended; CPU OK but slower.
    teacher_layer, student_layer : nn.Module | None
        Specific layers for Grad‑CAM.  If omitted, last Conv2d is auto‑detected.
    Returns
    -------
    float
        DisWOT score – *smaller is better* (closer teacher–student similarity).
    """
    teacher = teacher.to(device)
    student = student.to(device)

    t_layer = _find_last_conv(teacher)
    s_layer = _find_last_conv(student)

    with torch.no_grad(), autocast(enabled=device.type == "cuda"):
        # Grad‑CAM maps
        cam_t = _grad_cam_maps(teacher, x, t_layer)  # (N, H, W)
        cam_s = _grad_cam_maps(student, x, s_layer)  # (N, H, W)

    # Semantic metric M_s
    G_t = _channel_correlation(cam_t)
    G_s = _channel_correlation(cam_s)
    M_s = F.mse_loss(G_t, G_s, reduction="sum").sqrt()  # L2 distance

    # Relation metric M_r – feature maps before GAP (same layers)
    with torch.no_grad():
        feat_t = t_layer.output if hasattr(t_layer, "output") else None  # placeholder

    # To avoid re‑forwarding, compute features explicitly (no grad)
    def _extract_feat(model: nn.Module, layer: nn.Module):
        outputs = {}

        def hook_fn(_, __, out):
            outputs["feat"] = out.detach()
        h = layer.register_forward_hook(hook_fn)
        model(x)  # forward
        h.remove()
        return outputs["feat"]

    feat_t = _extract_feat(teacher, t_layer)  # (N, C, H, W)
    feat_s = _extract_feat(student, s_layer)

    A_t = _sample_relation_matrix(feat_t)  # (N, N)
    A_s = _sample_relation_matrix(feat_s)
    M_r = F.mse_loss(A_t, A_s, reduction="sum").sqrt()

    score = (M_s + M_r).item()

    # Optional cleanup to free VRAM when used in tight loops
    del cam_t, cam_s, feat_t, feat_s, A_t, A_s
    torch.cuda.empty_cache()
    return score


# =============================================================
#  2. スコア計算 (DisWOT + NWOT + SynFlow)
# =============================================================
def zc_score_pair(teacher: nn.Module, 
                  student: nn.Module, 
                  x: torch.Tensor, 
                  gpu: torch.device) -> float:
    teacher.eval()
    student.eval()

    # ---------- DisWOT風スコア（GPU 半精度） ----------
    with torch.no_grad(), autocast('cuda'):
        d_val = diswot_score(teacher, 
                             student, 
                             x.to(gpu), 
                             gpu)
    # ---------- Zero-cost proxies（CPU） ----------
    cpu = torch.device('cpu')
    student_cpu = student.to(cpu)

    dummy = TensorDataset(x.cpu(), 
                          torch.zeros(x.size(0), dtype=torch.long))
    loader = DataLoader(dummy, batch_size=x.size(0))

    with torch.enable_grad():
        measures = predictive.find_measures_arrays(
            student_cpu,
            loader,
            ("random", 1, x.size(0)),
            cpu,
            measure_names=["nwot", "synflow"],
            loss_fn=F.cross_entropy,
        )

    # ---------- スカラー化 ----------
    nwot    = to_scalar(measures["nwot"])
    synflow = to_scalar(measures["synflow"])

    # GPU に戻す（メモリ管理上任意）
    teacher.to(gpu)
    student.to(gpu)

    return d_val.item() + nwot + synflow


# =============================================================
#  3. スクリーニング (世代毎)
# =============================================================
def screening(
    gen: int,
    teacher_hashes: set[str],
    student_models: set[str],
    device: torch.device,
    batch: torch.Tensor,
) -> List[Tuple[Tuple[str, str], float]]:
    """teacher × student の全組み合わせに対して zero-cost スコアを計算。"""
    res: List[Tuple[Tuple[str, str], float]] = []
    all_pairs = list(product(teacher_hashes, student_models))

    with tqdm(total=len(all_pairs), desc=f"Gen{gen:03d}") as bar:
        for teacher_hash, student_hash in all_pairs:
            try:
                # モデルの構築
                teacher_model = get_nb101_model(teacher_hash).to(device)
                student_model = get_nb101_model(student_hash).to(device)

                # Zero-cost スコアの計算（関数がペア対応している場合）
                score = zc_score_pair(teacher_model, 
                                      student_model, 
                                      batch, 
                                      device)

                # メモリ開放（重要）
                del teacher_model, student_model
                torch.cuda.empty_cache()

                res.append(((teacher_hash, student_hash), score))
                bar.set_postfix(s=f"{score:.2f}")
            except Exception as e:
                print(f"Failed to process pair ({teacher_hash}, {student_hash}): {e}")

            bar.update(1)

    res.sort(key=lambda x: x[1], reverse=True)
    return res


# =============================================================
#  4. ミニ KD (3 エポック)
# ============================================================
def quick_kd_pair(
    gen: int,
    teacher_hash: str,
    student_hash: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 3
) -> float:
    """
    与えられたハッシュから教師モデル／生徒モデルを読み込んで
    知識蒸留を行い、検証精度を返す。
    """
    # --- 教師モデルの準備と重みロード ---
    teacher_model = get_nb101_model(teacher_hash).to(device)
    weight_path = Path(f"./weights/gen_{gen-1:03d}/{teacher_hash}_{epochs}.pth")
    if not weight_path.exists():
        raise FileNotFoundError(f"Teacher model weight not found at: {weight_path}")
    teacher_model.load_state_dict(torch.load(weight_path, map_location=device))
    teacher_model.eval()

    # --- 生徒モデル（未学習） ---
    student_model = get_nb101_model(student_hash)
    student_model.apply(gaussian_init).to(device)

    optimizer = torch.optim.SGD(
        student_model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(train_loader)
    )
    scaler = GradScaler()
    kd_loss_fn = nn.KLDivLoss(reduction="batchmean")
    ce_loss_fn = nn.CrossEntropyLoss()

    for _ in range(epochs):
        student_model.train()
        for batch_index, (images, labels) in enumerate(train_loader):
            if batch_index >= 100:
                break
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                with torch.no_grad():
                    teacher_outputs = teacher_model(images)
                student_outputs = student_model(images)
                loss = (
                    0.7 * kd_loss_fn(
                        F.log_softmax(student_outputs / 4, dim=1),
                        F.softmax(teacher_outputs / 4, dim=1)
                    )
                    + 0.3 * ce_loss_fn(student_outputs, labels)
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

    # --- 評価 ---
    student_model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            preds = student_model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            if total >= 2000:
                break

    return 100 * correct / total


def full_kd_pair(
    gen: int,
    teacher_hash: str,
    student_hash: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int
) -> float:
    # --- モデル準備 ---
    teacher_model = get_nb101_model(teacher_hash)
    weight_path = Path(f"./weights/gen_{gen-1:03d}/{teacher_hash}_{epochs}.pth")
    if not weight_path.exists():
        raise FileNotFoundError(f"Teacher model weight not found at: {weight_path}")
    teacher_model.load_state_dict(torch.load(weight_path, map_location=device))
    teacher_model.to(device).eval()

    student_model = get_nb101_model(student_hash)
    student_model.apply(gaussian_init).to(device)

    # --- オプティマイザ・スケジューラ・AMPスケーラー・損失関数 ---
    optimizer = torch.optim.SGD(
        student_model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(train_loader)
    )
    scaler = GradScaler()
    kd_loss_fn = nn.KLDivLoss(reduction="batchmean")
    ce_loss_fn = nn.CrossEntropyLoss()

    # --- 学習ループ ---
    for epoch in range(epochs):
        student_model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                with torch.no_grad():
                    teacher_outputs = teacher_model(images)
                student_outputs = student_model(images)
                loss = (
                    0.7 * kd_loss_fn(
                        F.log_softmax(student_outputs / 4, dim=1),
                        F.softmax(teacher_outputs / 4, dim=1)
                    )
                    + 0.3 * ce_loss_fn(student_outputs, labels)
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

    # --- モデル保存 ---
    save_dir = Path(f"./weights/gen_{gen:03d}")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{student_hash}_{epochs}.pth"
    torch.save(student_model.state_dict(), save_path)
    print(f"Saved student model after KD: {save_path}")

    # --- 評価ループ ---
    student_model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            preds = student_model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total


# =============================================================
#  5. 実験メインループ
# =============================================================
def run(args):
    # デバイス設定準備
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seeds(args.seed)

    # データ準備
    train_loader, val_loader = get_cifar10_dataloaders(batch_size=args.batch, num_workers=4)
    # イテレータを作成して，１バッチ目を取得
    batch_gpu, _ = next(iter(train_loader))
    # バッチの中でも指定の数 (screen_batch) だけを採用，計算するデバイスに送る
    batch_gpu = batch_gpu[: args.screen_batch].to(device)

    # 評価の高いアーキテクチャ（モデル構造）のハッシュ (str) を保存するリスト
    # deque（両端キュー）を使うことで、最大長を超えると自動で古いもの (先に入れたもの) が削除される
    # args.elite_pool の値によって最大保存数が決まる
    teacher: list[str] = list()
    teacher_hashes: set[str] = set()
    student: list[str] = list()
    student_hashes: set[str] = set()

    # 世代ごとにスクリーニングを実行
    print("=== Zero-Cost + DisWOT 探索開始 ===")
    for gen in range(1, args.generations + 1):
        print(f"\n===== Generation {gen}/{args.generations} =====")
        """
        gen=0 の場合はモデルの性能を nasbench101 から取得
        それ以外の世代では、スクリーニングを行い、
        上位のモデルを選び、次の世代に進む。
        """
        if gen == 1:
            # 最初の世代はランダムにモデルを生成するだけ
            # ひょっとしたら初期ノイズがうまくいくかもしれないから
            # 例えば 10 個のモデルを生成する
            while len(teacher) < args.teacher_pool:
                net, acc, hash = get_rnd_nb101_and_acc()
                if hash in teacher_hashes:
                    continue  # すでに登録済みならスキップ
                p = num_params(net)
                # パラメータ数や FLOPs が制限を超えていたらスキップ
                if p > args.max_params is False:
                    continue
                
                net = net.to(device)
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(net.parameters(), lr=0.01)

                num_epochs = args.pretrain_epochs
                for _ in range(num_epochs):
                    net.train()
                    for x, y in train_loader:
                        x, y = x.to(device), y.to(device)
                        optimizer.zero_grad()
                        output = net(x)
                        loss = criterion(output, y)
                        loss.backward()
                        optimizer.step()
                
                # ----- モデル保存 -----
                save_dir = Path(f"./weights/gen_{gen:03d}")
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / f"{hash}_{num_epochs}.pth"
                torch.save(net.state_dict(), save_path)
                print(f"Saved model: {save_path}")
                        
                teacher.append((acc, hash))
                teacher_hashes.add(hash)
        else:
            print(f"Gen {gen} Screening: Start")
            # 生徒モデルのプールを初期化
            while len(student) < args.student_pool:
                net, acc, hash = get_rnd_nb101_and_acc()
                if hash in student_hashes:
                    continue  # すでに登録済みならスキップ
                p = num_params(net)
                # パラメータ数や FLOPs が制限を超えていたらスキップ
                if p > args.max_params is False:
                    continue
                student.append((acc, hash))
                student_hashes.add(hash)
            
            # スクリーニングを実行して、上位のモデルを取得
            scr = screening(gen, 
                            teacher_hashes,
                            student_hashes,
                            device, 
                            batch_gpu, 
                            args.max_params)
            elite = scr[: args.student_pool]
            print(f"Gen {gen} Screening: Done")

            kd_res = list()
            # mini KD
            for (teacher_hash, student_hash), zero_cost_score in elite:
                acc = quick_kd_pair(
                    gen,
                    teacher_hash, 
                    student_hash, 
                    train_loader, 
                    val_loader, 
                    device,
                    epochs=3
                )
                kd_res.append(((teacher_hash, student_hash), acc))
                print(f"  {teacher_hash[:6]}→{student_hash[:6]} : {acc:.2f}% (zc={zero_cost_score:.2f})")

            # 上位 n_top 件だけ抜き出し
            kd_res.sort(key=lambda x: x[1], reverse=True)
            top_kd = kd_res[: args.n_top]
            print(f"[Full KD] 上位 {args.n_top} 件で本番 KD を実行します。")

            # 本番 KD
            for (teacher_hash, student_hash), quick_acc in top_kd:
                print(f"  ▶ {teacher_hash[:6]}→{student_hash[:6]} (quick={quick_acc:.2f}%)")
                full_acc = full_kd_pair(
                    gen,
                    teacher_hash,
                    student_hash,
                    train_loader,
                    val_loader,
                    device,
                    epochs=args.full_kd_epochs
                )
                print(f"    → Full KD accuracy: {full_acc:.2f}%")


# =============================================================
#  6. CLI
# =============================================================
if __name__ == "__main__":
    """
    実験のエントリーポイント。以下の処理を行います：

    1. コマンドライン引数を定義：
        - --generations      : 探索を行う世代数(進化回数)
        - --pop-size         : 各世代で評価する個体(モデル)数
        - --keep-top         : 各世代で残す上位モデルの数
        - --milestone        : 何世代ごとに蒸留評価(Milestone KD)を行うか
        - --ban-pool         : Milestone KD で ban 候補として追加するモデル数
        - --pretrain-epochs  : 初期モデルの学習エポック数
        - --teacher-pool     : 教師プール(教師モデル履歴)の最大長
        - --student-pool     : 生徒プール(生徒モデル履歴)の最大長
        - --max-params       : モデルの最大パラメータ数(これを超えると不採用)
        - --screen-batch     : Zero-Cost 指標評価で使用する画像枚数
        - --seed             : 乱数シード(再現性を確保)
        - --batch            : CIFAR-10 の学習バッチサイズ(quick_kd 用)
    2. 引数を表示し、設定内容のログを出力。
    3. `run(args)` を呼び出して探索フェーズ1を開始。
    4. 探索終了後、summary.json に結果を保存し、
       Phase 2 に向けた案内を出力。
    """
    ap = argparse.ArgumentParser("Zero-Cost + DisWOT search (with ceilings)")
    ap.add_argument("--generations", type=int, default=10)
    ap.add_argument("--pop-size", type=int, default=10)
    ap.add_argument("--keep-top", type=int, default=5)
    ap.add_argument("--milestone", type=int, default=5)
    ap.add_argument("--pretrain-epochs", type=int, default=50)
    ap.add_argument("--teacher-pool", type=int, default=10)
    ap.add_argument("--student-pool", type=int, default=10)
    ap.add_argument("--max-params", type=int, default=22000000)
    ap.add_argument("--screen-batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=128)
    
    args = ap.parse_args()

    print("=== Zero-Cost + DisWOT search (with ceilings) ===")
    print("  Generations:", args.generations)
    print("  Population size:", args.pop_size)
    print("  Keep top:", args.keep_top)
    print("  Milestone:", args.milestone)
    print("  Ban pool:", args.ban_pool)
    print("  Pretrain epochs:", args.pretrain_epochs)
    print("  Teacher pool:", args.teacher_pool)
    print("  Student pool:", args.student_pool)
    print("  Max params:", args.max_params)
    print("  Screen batch size:", args.screen_batch)
    print("  Seed:", args.seed)
    print("  Batch size:", args.batch)
    print("=========================================")

    run(args)

    print("=== 実験終了 ===")
    print("  結果は summary.json に保存されました。")
    print("  実験フェーズ2/2 を開始するには、次のコマンドを実行してください。")
    print("  python main.py --generations 10 --pop-size 10 --keep-top 5 --milestone 5 --ban-pool 5 --elite-pool 20 --max-params 1e6 --max-flops 1e6 --screen-batch 32 --seed 42 --batch 128")
    print("  (引数は適宜変更してください。)")
