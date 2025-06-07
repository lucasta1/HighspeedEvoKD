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
import gc
import csv
import argparse
import json
import math
import heapq
import random
import sys
import time
import copy
from tqdm import tqdm, trange
from pathlib import Path
from collections import deque
from itertools import product
from typing import Optional, List, Tuple
from fvcore.nn import FlopCountAnalysis
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import TensorDataset, DataLoader
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.max_workspace_size = 1024 * 1024**2   # 1 GiB
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import warnings
warnings.filterwarnings(
    "ignore",
    message=r"Detected call of `lr_scheduler.step\(\)` before `optimizer.step\(\)`",
    category=UserWarning
)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ---------- ユーザーパス調整 ----------
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT.parent))

# ---------- プロジェクト依存 ----------
from dataset.cifar10 import get_cifar10_dataloaders  # 32×32 データセット
from distiller_zoo import ICKDLoss, Similarity
# from models.nasbench101.build import (
#     get_nb101_model,
#     get_rnd_nb101_and_acc,
# )
from models.nasbench201.build import (
    get_nb201_model,
    get_rnd_nb201_and_acc,
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
# ---------- CUDA Prefetcher ----------
class CUDAPrefetcher:
    """次バッチをサブストリームで先に GPU 転送する薄いラッパー"""
    def __init__(self, loader: torch.utils.data.DataLoader, device: torch.device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream(device)
        self.iter_loader = iter(loader)
        self.next_data = None
        self.preload()

    def preload(self):
        try:
            data = next(self.iter_loader)
        except StopIteration:
            self.next_data = None
            return
        # ─ 非同期転送は別ストリームで ─
        with torch.cuda.stream(self.stream):
            if isinstance(data, (list, tuple)):
                self.next_data = tuple(
                    d.to(self.device, non_blocking=True) for d in data
                )
            else:
                self.next_data = data.to(self.device, non_blocking=True)

    def __iter__(self):
        self.iter_loader = iter(self.loader)  # <- ここで毎回初期化
        self.preload()
        return self

    def __next__(self):
        torch.cuda.current_stream(self.device).wait_stream(self.stream)
        if self.next_data is None:
            raise StopIteration
        data = self.next_data
        self.preload()
        return data
    
    def __len__(self):
        # DataLoader のバッチ数を返す
        return len(self.loader)


def set_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)


_MODEL_CACHE: dict[str, nn.Module] = {}
def get_model_cached(hash_str: str, device: torch.device) -> nn.Module:
    mdl = _MODEL_CACHE.get(hash_str)
    if mdl is None:
        mdl = get_nb201_model(hash_str).to(device)
        _MODEL_CACHE[hash_str] = mdl
    return mdl


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


def count_stride1_pool_layers(net):
    count = 0
    for m in net.modules():
        if isinstance(m, torch.nn.MaxPool2d):
            if m.stride == (1, 1) or m.stride == 1:
                count += 1
    return count


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

    # --- we don't need weight gradients ----------------------
    was_train = model.training
    model.eval()
    for p in model.parameters():       # freeze weights
        p.requires_grad_(False)

    x = x.detach().requires_grad_(True)  # only activation grads
    logits = model(x)
    score = logits.sum()
    score.backward()

    for p in model.parameters():       # restore flag
        p.requires_grad_(True)
    if was_train:
        model.train()

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

    with autocast(device_type="cuda", enabled=True):
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
    with torch.inference_mode():
        feat_s = _extract_feat(student, s_layer)

    A_t = _sample_relation_matrix(feat_t)  # (N, N)
    A_s = _sample_relation_matrix(feat_s)
    M_r = F.mse_loss(A_t, A_s, reduction="sum").sqrt()

    score = (M_s + M_r).item()

    # Optional cleanup to free VRAM when used in tight loops
    del cam_t, cam_s, feat_t, feat_s, A_t, A_s
    torch.cuda.empty_cache()
    return score


def check_net_configs(args: argparse.Namespace, net: nn.Module, device: torch.device) -> bool:
    p = num_params(net)
    # パラメータ数や FLOPs が制限を超えていたらスキップ
    if p > args.max_params:
        return False
    if count_stride1_pool_layers(net) > 3:
        # pool_count = count_stride1_pool_layers(net)
        # print(f"⚠️ Too many stride=1 pool layers ({pool_count}), skipping model.")
        return False
    dummy = torch.randn(1, 3, 32, 32).to(device)
    net = net.to(device)
    with torch.no_grad():
        out = net(dummy)
    net.to("cpu")
    del dummy
    torch.cuda.empty_cache()
    gc.collect()
    total_elements = out.numel()
    if total_elements > 1e6:
        #  print(f"⚠️ Output too large: {total_elements} elements")
        return False
    return True


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
    with autocast(device_type="cuda", enabled=True):
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

    return d_val # + nwot + synflow


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
                teacher_model = get_model_cached(teacher_hash, device) # .to(device)
                student_model = get_model_cached(student_hash, device) # .to(device)

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
#  4. Full KD
# ============================================================
def full_kd_pair(
    gen: int,
    teacher_hash: str,
    student_hash: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    timestamp: str,
    kd_weight: float = 0.7,
    ce_weight: float = 0.3,
) -> float:
    # --- モデル準備 ---
    teacher_model = get_nb201_model(teacher_hash).to(device)
    weight_path = Path(f"/mnt/newssd/weights_log/{timestamp}/gen_{gen-1:03d}/{teacher_hash}.pth")
    if not weight_path.exists():
        raise FileNotFoundError(f"Teacher model weight not found at: {weight_path}")
    teacher_model.load_state_dict(torch.load(weight_path, map_location=device))
    teacher_model.to(device).eval()

    student_model = get_nb201_model(student_hash).to(device)
    # student_model.apply(gaussian_init).to(device)

    # --- オプティマイザ・スケジューラ・AMPスケーラー・損失関数 ---
    optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(train_loader)
    )
    
    train_loader = CUDAPrefetcher(train_loader, device)   # ✔ 追加
    val_loader   = CUDAPrefetcher(val_loader,   device)
    
    kd_loss_fn = nn.KLDivLoss(reduction="batchmean")
    ce_loss_fn = nn.CrossEntropyLoss()

    # --- 学習ループ ---
    for epoch in range(epochs):
        student_model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                teacher_outputs = teacher_model(images)

            student_outputs = student_model(images)
            loss = (
                kd_weight * kd_loss_fn(
                    F.log_softmax(student_outputs / 4, dim=1),
                    F.softmax(teacher_outputs / 4, dim=1)
                )
                + ce_weight * ce_loss_fn(student_outputs, labels)
            )

            loss.backward()
            optimizer.step()
            scheduler.step()

    # --- モデル保存 ---
    save_dir = Path(f"/mnt/newssd/weights_log/{timestamp}/gen_{gen:03d}")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{student_hash}.pth"
    torch.save(student_model.state_dict(), save_path)
    # print(f"Saved student model after KD: {save_path}")

    # --- 評価ループ ---
    student_model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            preds = student_model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = 100 * correct / total
    
    teacher_model.to("cpu")
    student_model.to("cpu")
    del teacher_model, student_model
    torch.cuda.empty_cache()

    return acc


# =============================================================
#  5. 実験メインループ
# =============================================================
def print_gpu_mem(note=""):
    allocated = torch.cuda.memory_allocated() / 1024**2  # MB
    reserved = torch.cuda.memory_reserved() / 1024**2    # MB
    print(f"[GPU MEM] {note} | Allocated: {allocated:.1f} MB | Reserved: {reserved:.1f} MB")
    

def run(args):    
    # デバイス設定準備
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seeds(args.seed)
    timestamp = str(datetime.today().strftime("%Y%m%d_%H%M%S"))
    
    # プロファイリング
    profile_csv_path = f"timing_log_{timestamp}.csv"
    if not os.path.exists(profile_csv_path):
        with open(profile_csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Section", "Time(s)"])
            
    # acc の追跡
    acc_log = f"acc_log_{timestamp}.csv"
    if not os.path.exists(acc_log):
        with open(acc_log, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Generation", "Model ID", "Hash", "Accuracy"])
    
    ts_data_prep = time.perf_counter()  # 初期化（後で計測に使う）
    # データ準備
    # train_loader, val_loader = get_cifar10_dataloaders(batch_size=args.batch, num_workers=min(8, os.cpu_count()))
    train_loader_raw, val_loader_raw = get_cifar10_dataloaders(
        batch_size=args.batch,
        num_workers=min(8, os.cpu_count())
    )
    train_loader = CUDAPrefetcher(train_loader_raw, device)  # ✔ GPU プリフェッチ
    val_loader   = CUDAPrefetcher(val_loader_raw,   device)  # ✔ 同上
    # イテレータを作成して，１バッチ目を取得
    batch_gpu, _ = next(iter(train_loader))
    # バッチの中でも指定の数 (screen_batch) だけを採用，計算するデバイスに送る
    batch_gpu = batch_gpu[: args.screen_batch].to(device)
    tg_data_prep = time.perf_counter()
    with open(profile_csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Data Preparation", f"{tg_data_prep - ts_data_prep :.4f}"])
    

    # 評価の高いアーキテクチャ（モデル構造）のハッシュ (str) を保存するリスト
    # deque（両端キュー）を使うことで、最大長を超えると自動で古いもの (先に入れたもの) が削除される
    # args.elite_pool の値によって最大保存数が決まる
    # teacher: list[str] = list()
    teacher_hashes: set[str] = set()
    # student: list[str] = list()
    student_hashes: set[str] = set()

    # 世代ごとにスクリーニングを実行
    # ── Generation レベルのプログレスバー
    for gen in tqdm(range(1, args.generations+1), 
                    colour="blue", 
                    desc="🧬 Generations", 
                    position=0, 
                    leave=True):
        if gen == 1:
            # 最初の世代はランダムにモデルを生成するだけ (優秀なモデルに限定しない)
            start_time = time.perf_counter()
            model_id = 0
            while len(teacher_hashes) < args.teacher_pool:
                net, acc, hash = get_rnd_nb201_and_acc()
                if hash in teacher_hashes:
                    continue  # すでに登録済みならスキップ
                if not check_net_configs(args, net, device):
                    continue  # パラメータ数や FLOPs が制限を超えていたらスキップ
                
                ts_pretrain = time.perf_counter()  # 初期化（後で計測に使う）
                net = net.to(device)
                num_epochs = args.pretrain_epochs
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=num_epochs * len(train_loader)
                )
                # print(f"Total parameters: {sum(p.numel() for p in net.parameters()):,}")
                
                for epoch in tqdm(range(num_epochs),
                                  desc=f"Ancestor Pretraining",
                                  colour="green",
                                  position=1,
                                  leave=False):
                    net.train()
                    for x, y in train_loader:
                        x = x.to(device, 
                                 non_blocking=True,
                                 memory_format=torch.channels_last)
                        y = y.to(device, 
                                 non_blocking=True)
                        
                        optimizer.zero_grad()
                        output = net(x)
                        loss = criterion(output, y)
                        loss.backward()
                        optimizer.step()
                        scheduler.step()
                # 検証
                """
                correct = total = 0
                with torch.no_grad():
                    for images, labels in val_loader:
                        images, labels = images.to(device), labels.to(device)
                        preds = net(images).argmax(dim=1)
                        correct += (preds == labels).sum().item()
                        total += labels.size(0)
                acc = 100 * correct / total
                with open(acc_log, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([gen, model_id, hash, acc])
                """
                
                tg_pretrain = time.perf_counter()
                with open(profile_csv_path, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([f"Gen {gen} Pretrain", f"{tg_pretrain - ts_pretrain :.4f}"])
                    
                # ----- モデル保存 -----
                ts_savemodel_gen0 = time.perf_counter()
                save_dir = Path(f"/mnt/newssd/weights_log/{timestamp}/gen_{gen:03d}")
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / f"{hash}.pth"
                torch.save(net.state_dict(), save_path)
                print(f"Saved teacher model: {save_path}")
                
                net.to("cpu")               # パラメータを CPU へ退避
                del net, optimizer, criterion   # 参照を完全になくす
                torch.cuda.empty_cache()     # キャッシュ解放
                gc.collect()                 # Python ガベージコレクタ呼び出し
                
                # teacher.append((acc, hash))
                teacher_hashes.add(hash)
                end_time = time.perf_counter()
      
                tg_savemodel_gen0 = time.perf_counter()
                with open(profile_csv_path, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([f"Gen {gen} Save Model", f"{tg_savemodel_gen0 - ts_savemodel_gen0 :.4f}"])
                
                model_id += 1
                    
        else:
            start_time = time.perf_counter()
            # print(f"Gen {gen} Screening: Start")
            # 生徒モデルのプールを初期化
            while len(student_hashes) < args.student_pool:
                net, acc, hash = get_rnd_nb201_and_acc()
                if hash in teacher_hashes:
                    continue  # すでに登録済みならスキップ
                if not check_net_configs(args, net, device):
                    continue  # パラメータ数や FLOPs が制限を超えていたらスキップ
                # student.append((acc, hash))
                student_hashes.add(hash)
            # print(f"Gen {gen} Screening: Student generated")
            
            # スクリーニングを実行して，相性のいい教師と生徒のペアを取得
            # ── Screening レベルのプログレスバー
            all_pairs = list(product(teacher_hashes, student_hashes))
            scr = []
            pair_counter = 0
            for teacher_hash, student_hash in tqdm(all_pairs, 
                                                   desc="Screening",
                                                   total=len(all_pairs),
                                                   colour="yellow",
                                                   position=1,
                                                   leave=False):
                ts_pair_scoring = time.perf_counter()  # 初期化（後で計測に使う）
                # 元の screening 内の処理を呼び出し
                score = zc_score_pair(
                    get_model_cached(teacher_hash, device), # .to(device)
                    get_model_cached(student_hash, device), # .to(device)
                    batch_gpu,
                    device,
                )
                scr.append(((teacher_hash, student_hash), score))
                pair_counter += 1
                tg_pair_scoring = time.perf_counter()
                with open(profile_csv_path, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([f"Gen {gen} Pair Scoring {pair_counter}", f"{tg_pair_scoring - ts_pair_scoring :.4f}"])
                
            scr.sort(key=lambda x: x[1], reverse=True)
            elite = scr[: args.student_pool]
            # print(f"Gen {gen} Screening: Done")
            
            teacher_hashes = set()
            student_hashes = set()

            # print(f"[Full KD] 上位 {args.student_pool} 件で本番 KD を実行します。")
            # ── Full KD レベルのプログレスバー
            model_id
            for (t_hash, s_hash), quick_acc in tqdm(elite,
                                                    desc="Full KD",
                                                    total=len(elite),
                                                    colour="magenta",
                                                    position=2,
                                                    leave=False):
                ts_kd = time.perf_counter()  # 初期化（後で計測に使う）
                # 本番 KD 実行
                full_acc = full_kd_pair(
                    gen,
                    t_hash,
                    s_hash,
                    train_loader,
                    val_loader_raw,
                    device,
                    epochs=args.full_kd_epochs,
                    timestamp=timestamp,
                    kd_weight=args.kd_weight,
                    ce_weight=args.ce_weight,
                )
                teacher_hashes.add(s_hash)
                print(f"[Full KD] Gen {gen} {str(t_hash)[:6]}→{str(s_hash)[:6]} : {full_acc:.2f}%")
                tg_kd = time.perf_counter()
                with open(profile_csv_path, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([f"Gen {gen} Full KD {str(t_hash)[:6]}→{str(s_hash)[:6]}", f"{tg_kd - ts_kd :.4f}"])
                with open(acc_log, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([gen, model_id, s_hash, full_acc])
                model_id += 1
            end_time = time.perf_counter()
            print(f"[Gen {gen}] Full KD completed in {end_time - start_time:.2f} seconds")


# =============================================================
#  6. CLI
# =============================================================
if __name__ == "__main__":
    """
    実験のエントリーポイント。以下の処理を行います：

    1. コマンドライン引数を定義：
        - --generations      : 探索を行う世代数(進化回数)
        - --pretrain-epochs  : 初期モデルの学習エポック数
        - --full-kd-epochs   : 本番 KD の学習エポック数
        - --teacher-pool     : 教師プール(教師モデル履歴)の最大長
        - --student-pool     : 生徒プール(生徒モデル履歴)の最大長
        - --max-params       : モデルの最大パラメータ数(これを超えると不採用)
        - --screen-batch     : Zero-Cost 指標評価で使用する画像枚数
        - --seed             : 乱数シード(再現性を確保)
        - --batch            : CIFAR-10 の学習バッチサイズ(quick_kd 用)
        - --kd-weight        : 知識蒸留損失の重み
        - --ce-weight        : クロスエントロピー損失の重み
        
    2. 引数を表示し、設定内容のログを出力。
    3. `run(args)` を呼び出して探索フェーズ1を開始。
    4. 探索終了後、summary.json に結果を保存し、
       Phase 2 に向けた案内を出力。
    """
    ap = argparse.ArgumentParser("Zero-Cost + DisWOT search (with ceilings)")
    ap.add_argument("--generations", type=int, default=100)
    ap.add_argument("--pretrain-epochs", type=int, default=3)
    ap.add_argument("--full-kd-epochs", type=int, default=3)
    ap.add_argument("--teacher-pool", type=int, default=10)
    ap.add_argument("--student-pool", type=int, default=10)
    ap.add_argument("--max-params", type=int, default=10000000)
    ap.add_argument("--screen-batch", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--kd-weight", type=float, default=0.7, help="Weight for KD loss")
    ap.add_argument("--ce-weight", type=float, default=0.3, help="Weight for CE loss")
    
    args = ap.parse_args()

    print("=== Zero-Cost + DisWOT search (with ceilings) ===")
    print("  Generations:", args.generations)
    print("  Pretrain epochs:", args.pretrain_epochs)
    print("  Full KD epochs:", args.full_kd_epochs)
    print("  Teacher pool:", args.teacher_pool)
    print("  Student pool:", args.student_pool)
    print("  Max params:", args.max_params)
    print("  Screen batch size:", args.screen_batch)
    print("  Seed:", args.seed)
    print("  Batch size:", args.batch)
    print("  KD weight:", args.kd_weight)
    print("  CE weight:", args.ce_weight)
    print("=========================================")

    run(args)

    print("=== 実験終了 ===")
    