import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zscore

BASE_DIR = r"dataset/generation/gen/images"
OUTPUT_DIR = r"dataset/generation/gen/stats_similarity"

RESULTS_FILE = os.path.join(
    OUTPUT_DIR,
    "similarity_results.csv"
)

df = pd.read_csv(RESULTS_FILE)

# ==========================================================
# REBUILD SUMMARY
# ==========================================================

summary = (
    df.groupby("model")
      .agg({
          "unique_identities": "mean",
          "mean_nn": "mean",
          "largest_cluster": "mean",
          "pairs_085": "mean",
          "pairs_090": "mean",
      })
      .sort_values(
          "unique_identities",
          ascending=False,
      )
)

summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "similarity_summary.csv"
    )
)

# ==========================================================
# REBUILD DIVERSITY SCORE
# ==========================================================

score_df = df.copy()
score_df["z_identities"] = zscore(
    score_df["unique_identities"]
)
score_df["z_mean_nn"] = zscore(
    score_df["mean_nn"]
)
score_df["z_085"] = zscore(
    score_df["pairs_085"]
)
score_df["z_090"] = zscore(
    score_df["pairs_090"]
)
score_df["diversity_score"] = (
    score_df["z_identities"]
    - score_df["z_mean_nn"]
    - score_df["z_085"]
    - score_df["z_090"]
)

ranking = (
    score_df.groupby("model")
    .agg({
        "diversity_score": "mean",
        "unique_identities": "mean",
        "mean_nn": "mean",
        "pairs_085": "mean",
        "pairs_090": "mean",
    })
    .sort_values(
        "diversity_score",
        ascending=False,
    )
)

ranking.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "diversity_ranking.csv"
    )
)

# ==========================================================
# GRAPH HELPER
# ==========================================================

def save_line_plot(
    column,
    title,
    ylabel,
    filename,
):

    plt.figure(figsize=(10, 6))

    for model in sorted(df["model"].unique()):

        subset = df[
            df["model"] == model
        ].sort_values("age")

        plt.plot(
            subset["age"],
            subset[column],
            marker="o",
            linewidth=2,
            label=model,
        )

    plt.title(title)
    plt.xlabel("Age")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            filename
        ),
        dpi=300,
    )

    plt.close()

# ==========================================================
# REBUILD PLOTS
# ==========================================================

save_line_plot(
    "unique_identities",
    "Unique Identities by Age",
    "Unique Identities",
    "unique_identities.png",
)

save_line_plot(
    "mean_similarity",
    "Mean Similarity by Age",
    "Similarity",
    "mean_similarity.png",
)

save_line_plot(
    "mean_nn",
    "Mean Nearest Neighbor Similarity by Age",
    "Similarity",
    "mean_nearest_neighbor.png",
)

save_line_plot(
    "largest_cluster",
    "Largest Identity Cluster by Age",
    "Cluster Size",
    "largest_cluster.png",
)

save_line_plot(
    "pairs_080",
    "Pairs >= 0.80 by Age",
    "Pair Count",
    "pairs_080.png",
)

save_line_plot(
    "pairs_085",
    "Pairs >= 0.85 by Age",
    "Pair Count",
    "pairs_085.png",
)

save_line_plot(
    "pairs_090",
    "Pairs >= 0.90 by Age",
    "Pair Count",
    "pairs_090.png",
)
# ==========================================================
# SUMMARY TABLE IMAGE
# ==========================================================

fig, ax = plt.subplots(
    figsize=(10, max(3, len(summary) * 0.5))
)

ax.axis("off")

table = ax.table(
    cellText=np.round(
        summary.values,
        3
    ),
    colLabels=summary.columns,
    rowLabels=summary.index,
    loc="center",
)

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.5)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "similarity_summary.png"
    ),
    dpi=300,
    bbox_inches="tight",
)

plt.close()
# ==========================================================
# RANKING TABLE IMAGE
# ==========================================================

fig, ax = plt.subplots(
    figsize=(10, max(3, len(ranking) * 0.5))
)

ax.axis("off")

table = ax.table(
    cellText=np.round(
        ranking.values,
        3
    ),
    colLabels=ranking.columns,
    rowLabels=ranking.index,
    loc="center",
)

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.5)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "diversity_ranking.png"
    ),
    dpi=300,
    bbox_inches="tight",
)

plt.close()
print("Done.")