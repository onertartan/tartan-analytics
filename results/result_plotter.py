import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re

DATA_DIR = Path("files")

files = sorted(DATA_DIR.glob("spectral_*.csv"))
if not files:
    raise FileNotFoundError("No spectral CSV files found in ")
print("ÇŞL")
def parse_filename(fname):
    """
    Expected format:
    spectral_<SCALER NAME>_<N_NEIGHBORS>.csv

    Examples:
    spectral_Share of Top 30 (L1 Norm)_20.csv
    spectral_Share of Total_7.csv
    """
    m = re.match(r"spectral_(.+)_(\d+)\.csv", fname.name)
    if m is None:
        raise ValueError(f"Cannot parse filename: {fname.name}")
    scaler = m.group(1)
    n_neighbors = int(m.group(2))
    return scaler, n_neighbors

dfs = []
for f in files:
    scaler, n_nb = parse_filename(f)
    df = pd.read_csv(f, index_col=0)
    df["k"] = df.index.astype(int)
    df["scaler"] = scaler
    df["n_neighbors"] = n_nb
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re

# ... (parse_filename ve veri yükleme kısımları aynı) ...

scalers = list(dict.fromkeys(df_all["scaler"]))
n_rows = len(scalers) + 1
fig, axes = plt.subplots(
    nrows=n_rows,
    ncols=3,
    figsize=(15, 4 * n_rows),
    sharex=True
)

metrics = [
    ("Silhouette_mean", "Silhouette (cosine)", "Separation"),
    ("ARI_mean", "Mean ARI", "Stability"),
    ("Consensus", "Consensus index", "Dominant structure"),
]

# Kısaltmalar ve Renkler
scaler_abbr = {
    "Share of Top 30 (L1 Norm)": "L1",
    "Share of Total": "S",
    "TF-IDF": "TF",
    "L2 Normalization": "L2"
}
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Her scaler için sabit renk

for j, (col, ylabel, title) in enumerate(metrics):
    # --- İLK 4 SATIR: HER BİR SCALER DETAYI ---
    for i, scaler in enumerate(scalers):
        ax = axes[i, j]
        df_s = df_all[df_all["scaler"] == scaler]

        for n_nb, g in df_s.groupby("n_neighbors"):
            ax.plot(g["k"], g[col], linewidth=1, alpha=0.3)

        # Her scaler'ın kendi içindeki en iyilerini etiketle
        best_indices = df_s.groupby("k")[col].idxmax()
        df_best = df_s.loc[best_indices]
        for _, row in df_best.iterrows():
            ax.text(row["k"], row[col], f'{int(row["n_neighbors"])}',
                    fontsize=8, fontweight='bold', va='bottom', ha='center', color='black')

        # Eksen ve başlık ayarları
        if i == 0: ax.set_title(title, fontsize=11)
        if j == 0:
            ax.set_ylabel(f"{ylabel}\n[{scaler}]")
        else:
            ax.set_ylabel(ylabel)

        if col in ("ARI_mean", "Consensus"): ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.2)

    # --- 5. SATIR: TÜM SCALER'LARIN EN İYİLERİ (4 ÇİZGİ) ---
    # --- 5. SATIR: TÜM SCALER'LARIN EN İYİLERİ ---
    ax_comp = axes[-1, j]

    # Her k noktası için bilgileri ve o k'daki maksimum y değerini saklayacağız
    k_data = {k: {'infos': [], 'max_y': 0} for k in df_all["k"].unique()}

    # 1. Verileri topla ve her k için o sütundaki en yüksek noktayı belirle
    for i, scaler in enumerate(scalers):
        df_s = df_all[df_all["scaler"] == scaler]
        idx = df_s.groupby("k")[col].idxmax()
        df_scaler_best = df_s.loc[idx].sort_values("k")

        abbr = scaler_abbr.get(scaler, "")
        color = colors[i % len(colors)]

        # Çizgiyi çiz
        line, = ax_comp.plot(
            df_scaler_best["k"],
            df_scaler_best[col],
            'o-',
            linewidth=2,
            label=abbr,
            color=color,
            alpha=0.7
        )

        for _, row in df_scaler_best.iterrows():
            k_val = row["k"]
            val = row[col]
            k_data[k_val]['infos'].append((f"{abbr}:{int(row['n_neighbors'])}", color))
            # O sütundaki en yüksek y değerini güncelle
            if val > k_data[k_val]['max_y']:
                k_data[k_val]['max_y'] = val

    # 2. Etiketleri her sütunun zirve noktasının 30 piksel üzerinden başlayarak alt alta yazdır
    for k_val, data in k_data.items():
        peak_y = data['max_y']

        for rank, (text, color) in enumerate(data['infos']):
            # rank 0: 30px üstte, rank 1: 38px üstte, rank 2: 46px üstte...
            y_offset = 20 + (rank * 8)
            if k_val<=3 or j==1:
                y_offset-=10
            ax_comp.annotate(
                text,
                xy=(k_val, peak_y),
                xytext=(0, y_offset),
                textcoords="offset points",
                fontsize=5,
                fontweight='bold',
                color=color,
                ha='left',
                va='bottom',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.2)
            )

    # Grafik tavanını etiketlere yer açmak için %15-20 oranında yükseltelim
    current_ylim = ax_comp.get_ylim()
    ax_comp.set_ylim(current_ylim[0], current_ylim[1] * 1.1)

    ax_comp.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1, 1))  # Lejantı dışarı alabiliriz
    ax_comp.grid(True, alpha=0.2)

# Tüm x ekseni ayarları
for ax in axes.flat:
    ax.set_xticks(sorted(df_all["k"].unique()))
    ax.set_xlabel("Number of clusters (k)")
print("zxcasd",len(axes))
plt.tight_layout()
plt.show()
"""
SELECTED_K = 6
for row in axes:
    for ax in row:
        ax.axvline(
            SELECTED_K,
            color="black",
            linestyle="--",
            alpha=0.5
        )
"""

