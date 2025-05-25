import pandas as pd
import matplotlib.pyplot as plt
import os

# データ読み込み
csv_path = "param_flops_stats.csv"
df = pd.read_csv(csv_path)
df_flops = df[df["flops"] > 0]

# 出力ディレクトリ（任意で変更）
output_dir = "output_graphs"
os.makedirs(output_dir, exist_ok=True)


# ヘルパー関数
def plot_hist_with_percentiles(data, column, title, unit="M", filename="output.png"):
    values = data[column] / 1e6  # 単位変換

    p90 = values.quantile(0.9)
    p95 = values.quantile(0.95)

    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=30, color='skyblue', edgecolor='black')
    plt.title(f"{title} Distribution")
    plt.xlabel(f"{title} ({unit})")
    plt.ylabel("Count")

    for p, label in zip([p90, p95], ["p90", "p95"]):
        plt.axvline(p, color='red', linestyle='--', label=f"{label} = {p:.2f} {unit}")
        plt.text(p, plt.ylim()[1]*0.9, f"{label}: {p:.2f} {unit}", color='red', rotation=90, va='top')

    plt.legend()
    plt.tight_layout()

    # 保存
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path)
    plt.close()  # メモリ節約のため閉じる


# 保存実行（params）
plot_hist_with_percentiles(df, "params", "Model Parameters", filename="params_hist.png")


# 保存実行（flops）
if not df_flops.empty:
    plot_hist_with_percentiles(df_flops, "flops", "Model FLOPs", filename="flops_hist.png")

