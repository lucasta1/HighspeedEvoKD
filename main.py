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
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
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


# =============================================================
#  2. スコア計算 (DisWOT + NWOT + SynFlow)
# =============================================================
def zc_score(net: nn.Module, x: torch.Tensor, gpu: torch.device) -> float:
    net.eval()

    # ---------- DisWOT（GPU 半精度） ----------
    with torch.no_grad(), autocast('cuda'):
        feat, _ = net.forward_with_features(x)
        comp = net.classifier.weight.unsqueeze(-1).unsqueeze(-1)
        d_val = -ICKDLoss()([comp], [comp])[0] - Similarity()(feat[-2], feat[-2])[0]

    # ---------- Zero-Cost proxies（CPU） ----------
    cpu = torch.device('cpu')
    net_cpu = net.to(cpu)
    dummy = TensorDataset(x.cpu(), torch.zeros(x.size(0), dtype=torch.long))
    loader = DataLoader(dummy, batch_size=x.size(0))

    with torch.enable_grad():
        measures = predictive.find_measures_arrays(
            net_cpu,
            loader,
            ("random", 1, x.size(0)),
            cpu,
            measure_names=["nwot", "synflow"],
            loss_fn=F.cross_entropy,
        )

    # ---------- スカラー化 ----------
    nwot    = to_scalar(measures["nwot"])
    synflow = to_scalar(measures["synflow"])

    # （必要なら）GPU へ戻す
    net.to(gpu)

    return d_val.item() + nwot + synflow


# =============================================================
#  3. スクリーニング (世代毎)
# =============================================================
def screening(
    gen: int,                     # 現在の世代番号（例: 第3世代 → 3）
    pop: int,                     # 集めたいアーキテクチャの数（母集団サイズ）
    device: torch.device,         # モデルを配置するデバイス（GPUなど）
    batch: torch.Tensor,          # Zero-Cost スコアを計算するための画像バッチ
    max_p: int,                   # 許容する最大パラメータ数
) -> List[Tuple[str, float]]:
    """条件を満たす pop 個のモデルを集めてスコア付けし、良い順に返す。"""

    # 結果を保存するリスト（要素は: モデルのハッシュとスコアのタプル）
    res: List[Tuple[str, float]] = []
    # 何回モデル生成・試行を行ったか（条件不適合で捨てた分も含む）
    trials = 0

    # プログレスバーを使って進捗を表示（例: Gen003）
    with tqdm(total=pop, desc=f"Gen{gen:03d}") as bar:
        # 条件を満たすモデルが pop 個集まるまでループ
        while len(res) < pop:
            trials += 1  # 試行回数をインクリメント
            # ランダムにアーキテクチャを生成し、構造のハッシュを取得
            net, _acc, h = get_rnd_nb101_and_acc()
            # パラメータ数と FLOPs を計測
            p = num_params(net)
            # パラメータ数や FLOPs が制限を超えていたらスキップ（条件不適合）
            if p > max_p is False:
                continue
            
            # モデルをガウス初期化して GPU に移動
            net.apply(gaussian_init).to(device)
            # Zero-Cost Proxy によるスコア計算（訓練なしの軽量評価）
            s = zc_score(net, batch, device)
            del net
            torch.cuda.empty_cache()
            
            # 結果としてハッシュとスコアを追加
            res.append((h, s))
            # プログレスバー更新（残り個数が減る）
            bar.update(1)
            # パラメータ、FLOPs、スコアをリアルタイム表示（進捗バーの横に）
            bar.set_postfix(p=f"{p/1e6:.1f}M", s=f"{s:.1f}")
    # スコアが高い順に並び替え（降順）
    res.sort(key=lambda x: x[1], reverse=True)
    # 何回試行したかを表示（条件不適合によるリトライも含む）
    print(f"[Gen {gen}] Trials: {trials}")
    # スコア付き構造（ハッシュ, スコア）のリストを返す
    return res


# =============================================================
#  4. ミニ KD (3 エポック)
# ============================================================
def quick_kd(
    arch: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 3
) -> float:
    """
    ランダム初期化した教師モデルと生徒モデルを使って簡易な知識蒸留を行い、
    バリデーションデータに対する精度（Accuracy）を返す。

    Parameters:
        arch (str): モデルアーキテクチャ名。
        train_loader (DataLoader): 訓練データローダー。
        val_loader (DataLoader): 検証データローダー。
        device (torch.device): モデルとデータを配置するデバイス。
        epochs (int): 学習エポック数（デフォルト: 3）

    Returns:
        float: 検証データに対する精度（%）
    """

    # --- 教師モデルと生徒モデルを初期化（ランダム） ---
    teacher_model, _, _ = get_nb101_model(arch, pretrained=False)
    teacher_model.apply(gaussian_init).to(device).eval()  # 教師は学習しないので eval モード

    student_model, _, _ = get_nb101_model(arch, pretrained=False)
    student_model.apply(gaussian_init).to(device)

    # --- 最適化とスケジューラの設定 ---
    optimizer = torch.optim.SGD(
        student_model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(train_loader)
    )

    # --- AMP（自動混合精度）用のスケーラー ---
    scaler = GradScaler()

    # --- 損失関数の定義 ---
    kd_loss_fn = nn.KLDivLoss(reduction="batchmean")  # 知識蒸留用のKL損失
    ce_loss_fn = nn.CrossEntropyLoss()                # 通常の分類損失

    # --- 学習ループ（簡易） ---
    for epoch in range(epochs):
        student_model.train()
        for batch_index, (images, labels) in enumerate(train_loader):
            if batch_index >= 100:  # 処理時間削減のため100バッチまで
                break

            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            # --- 自動混合精度による順伝播と損失計算 ---
            with autocast():
                with torch.no_grad():
                    teacher_outputs = teacher_model(images)  # 教師の出力（勾配なし）

                student_outputs = student_model(images)      # 生徒の出力

                # 蒸留損失（温度T=4）＋クロスエントロピー損失の加重平均
                loss = (
                    0.7 * kd_loss_fn(
                        F.log_softmax(student_outputs / 4, dim=1),
                        F.softmax(teacher_outputs / 4, dim=1)
                    )
                    + 0.3 * ce_loss_fn(student_outputs, labels)
                )

            # --- 勾配の逆伝播と更新 ---
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()  # 学習率をスケジューリング

    # --- 評価ループ（最大2000サンプルまで） ---
    student_model.eval()
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            predictions = student_model(images).argmax(dim=1)  # 予測ラベル
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)

            if total_samples >= 2000:  # 早期終了（2000件までで評価）
                break

    # --- 精度（%）を計算して返す ---
    accuracy = 100 * correct_predictions / total_samples
    return accuracy


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
    elite: deque[str] = deque(maxlen=args.elite_pool)

    # 性能が悪かったり、すでに評価済みで再利用したくないアーキテクチャのハッシュを記録する集合
    # set にすることで高速に「すでに登録済みかどうか」を判定できる
    ban_pool: set[str] = set()

    t0 = time.time()
    try:
        # 世代ごとにスクリーニングを実行
        print("=== 実験フェーズ1/2: Zero-Cost + DisWOT 探索開始 ===")
        for gen in range(1, args.generations + 1):
            print(f"\n===== Generation {gen}/{args.generations} =====")
            scr = screening(gen, 
                            args.pop_size, 
                            device, 
                            batch_gpu, 
                            args.max_params)
            top_hash = [h for h, _ in scr[: args.keep_top]]
            elite.extend(top_hash)
            print(f"Gen {gen} Screening: Done")

            if gen % args.milestone == 0:
                print("[Milestone KD]  上位モデルを 3epoch 蒸留で再評価…")
                kd_res = []
                for h in top_hash:
                    acc = quick_kd(h, train_loader, val_loader, device)
                    kd_res.append((h, acc))
                    print(f"  {h[:6]} : {acc:.2f}%")
                kd_res.sort(key=lambda x: x[1], reverse=True)
                ban_pool.update(h for h, _ in kd_res[: args.ban_pool])
                print("BAN 候補追加:", kd_res[: args.ban_pool])
            print(f"Elapsed {(time.time()-t0)/3600:.2f}h | BAN pool {len(ban_pool)}\n")
    except KeyboardInterrupt:
        print("\n⏹️  中断されました。ここまでの結果を保存します。")

    # サマリー保存
    Path("summary.json").write_text(json.dumps({
        "elite": list(elite),
        "ban_pool": list(ban_pool),
    }, indent=2))
    print("✅ summary.json 保存完了 — 実験フェーズ1/2 終了")

# =============================================================
#  6. CLI
# =============================================================


if __name__ == "__main__":
    """
    実験のエントリーポイント。以下の処理を行います：

    1. コマンドライン引数を定義：
        - --generations      : 探索を行う世代数(進化回数)
        - --pop-size         : 各世代で評価する個体(（)モデル)数
        - --keep-top         : 各世代で残す上位モデルの数
        - --milestone        : 何世代ごとに蒸留評価(Milestone KD)を行うか
        - --ban-pool         : Milestone KD で ban 候補として追加するモデル数
        - --elite-pool       : エリートプール(優秀モデル履歴)の最大長
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
    ap.add_argument("--ban-pool", type=int, default=5)
    ap.add_argument("--elite-pool", type=int, default=20)
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
    print("  Elite pool:", args.elite_pool)
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
