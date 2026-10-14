import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from deepface import DeepFace
from scipy.stats import zscore
# ==========================================================
# CONFIG
# ==========================================================

BASE_DIR = r"dataset/generation/gen/images"
OUTPUT_DIR = r"dataset/generation/gen/stats_similarity"

MODELS = sorted(
    folder
    for folder in os.listdir(BASE_DIR)
    if os.path.isdir(os.path.join(BASE_DIR, folder))
)

EMBEDDING_MODEL = "Facenet512"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

IDENTITY_THRESHOLD = 0.85

# ==========================================================
# UNION-FIND
# ==========================================================

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


# ==========================================================
# ANALYSIS
# ==========================================================

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

NUM_THREADS = 4

def load_embedding(path):
    try:
        result = DeepFace.represent(
            img_path=path,
            model_name=EMBEDDING_MODEL,
            detector_backend="skip",
            enforce_detection=False,
        )
        return np.array(
            result[0]["embedding"],
            dtype=np.float32,
        )
    except Exception:
        return None


os.makedirs(OUTPUT_DIR, exist_ok=True)
results = []

for model in MODELS:
    model_dir = os.path.join(BASE_DIR, model)

    ages = sorted(
        int(folder)
        for folder in os.listdir(model_dir)
        if folder.isdigit()
        and os.path.isdir(os.path.join(model_dir, folder))
    )

    for age in tqdm(ages, desc=f"{model}", leave=True):

        image_dir = os.path.join(model_dir, str(age))

        image_paths = []

        for file in sorted(os.listdir(image_dir)):
            ext = os.path.splitext(file)[1].lower()

            if ext in IMAGE_EXTENSIONS:
                image_paths.append(
                    os.path.join(image_dir, file)
                )

        with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            embeddings = [
                e
                for e in tqdm(
                    executor.map(load_embedding, image_paths),
                    total=len(image_paths),
                    desc=f"{model} age {age} embeddings",
                    leave=False,
                )
                if e is not None
            ]

        n = len(embeddings)

        if n < 2:
            continue

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        embeddings /= np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True,
        )

        uf = UnionFind(n)

        # --------------------------------------------------
        # Similarity matrix (computed once)
        # --------------------------------------------------

        sim_matrix = embeddings @ embeddings.T

        # nearest-neighbor should ignore self similarity
        np.fill_diagonal(sim_matrix, -1.0)

        nearest_neighbor = sim_matrix.max(axis=1)

        # Only unique pairs (i < j)
        upper_i, upper_j = np.triu_indices(n,k=1,)

        pairwise_scores = sim_matrix[upper_i,upper_j,]

        pairs_080 = np.count_nonzero(pairwise_scores >= 0.80)
        pairs_085 = np.count_nonzero(pairwise_scores >= 0.85)
        pairs_090 = np.count_nonzero(pairwise_scores >= 0.90)

        # --------------------------------------------------
        # Identity clustering
        # --------------------------------------------------

        match_indices = np.where(
            pairwise_scores >= IDENTITY_THRESHOLD
        )[0]

        for idx in match_indices:
            uf.union(
                upper_i[idx],
                upper_j[idx],
            )

        # --------------------------------------------------
        # Cluster stats
        # --------------------------------------------------

        clusters = {}

        for idx in range(n):
            root = uf.find(idx)
            if root not in clusters:
                clusters[root] = []
            clusters[root].append(idx)

        cluster_sizes = sorted((len(v) for v in clusters.values()), reverse=True,)

        identity_count = len(clusters)

        results.append({
            "model": model,
            "age": age,
            "faces": n,
            "mean_similarity": float(pairwise_scores.mean()),
            "median_similarity": float(np.median(pairwise_scores)),
            "max_similarity": float(pairwise_scores.max()),
            "mean_nn": float(nearest_neighbor.mean()),
            "median_nn": float(np.median(nearest_neighbor)),
            "max_nn": float(nearest_neighbor.max()),
            "unique_identities": identity_count,
            "largest_cluster": max(cluster_sizes),
            "pairs_080": int(pairs_080),
            "pairs_085": int(pairs_085),
            "pairs_090": int(pairs_090),
        })

# ==========================================================
# DATAFRAMES
# ==========================================================

df = pd.DataFrame(results)
if len(df) == 0:
    raise RuntimeError("No valid results generated.")
df = df.sort_values(
    ["model", "age"]
)
df.to_csv(os.path.join(OUTPUT_DIR, "similarity_results.csv"), index=False)
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
summary.to_csv(os.path.join(OUTPUT_DIR, "similarity_summary.csv"))
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
# PLOT HELPER
# ==========================================================

plt.style.use("default")
def save_line_plot(
    column,
    title,
    ylabel,
    filename,
):
    plt.figure(figsize=(10, 6))

    for model in MODELS:
        subset = df[
            df["model"] == model
        ]
        if len(subset) == 0:
            continue
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
        os.path.join(OUTPUT_DIR, filename),
        dpi=300,
    )
    plt.close()


# ==========================================================
# LINE CHARTS
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
# HEATMAPS (optional)
# ==========================================================

try:
    import seaborn as sns
    heatmaps = [
        (
            "unique_identities",
            "Unique Identities",
            "heatmap_unique_identities.png",
        ),
        (
            "mean_nn",
            "Mean Nearest Neighbor Similarity",
            "heatmap_mean_nn.png",
        ),
        (
            "largest_cluster",
            "Largest Identity Cluster",
            "heatmap_largest_cluster.png",
        ),
    ]

    for metric, title, filename in heatmaps:
        pivot = df.pivot(
            index="model",
            columns="age",
            values=metric,
        )
        plt.figure(
            figsize=(
                max(8, len(pivot.columns)),
                max(4, len(pivot.index) * 0.6),
            )
        )
        sns.heatmap(
            pivot,
            annot=True,
            cmap="viridis",
            fmt=".2f",
        )
        plt.title(title)
        plt.tight_layout()
        plt.savefig(
            os.path.join(OUTPUT_DIR, filename),
            dpi=300,
        )
        plt.close()
except ImportError:
    pass

# ==========================================================
# COMPOSITE DIVERSITY SCORE
# ==========================================================

score_df = df.copy()
score_df["z_identities"] = zscore(
    score_df["unique_identities"]
)

score_df["z_mean_nn"] = zscore(
    score_df["mean_nn"]
)

score_df["z_cluster"] = zscore(
    score_df["largest_cluster"]
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
    - score_df["z_cluster"]
    - score_df["z_085"]
    - score_df["z_090"]
)
plt.figure(figsize=(10, 6))

ages = sorted(df["age"].unique())


plt.figure(figsize=(10, 6))

for model in MODELS:
    subset = score_df[
        score_df["model"] == model
    ]
    if len(subset) == 0:
        continue
    plt.plot(
        subset["age"],
        subset["diversity_score"],
        marker="o",
        linewidth=2,
        label=model,
    )

plt.title("Composite Diversity Score")
plt.xlabel("Age")
plt.ylabel("Score")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "diversity_score.png"),
    dpi=300,
)
plt.close()

# ==========================================================
# MODEL RANKING CSV
# ==========================================================

ranking = (
    score_df.groupby("model")
    .agg({
        "diversity_score": "mean",
        "unique_identities": "mean",
        "mean_nn": "mean",
        "largest_cluster": "mean",
        "pairs_085": "mean",
        "pairs_090": "mean",
    })
    .sort_values(
        "diversity_score",
        ascending=False,
    )
)
ranking.to_csv(os.path.join(OUTPUT_DIR, "diversity_ranking.csv"))
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

print("Analysis complete.")
print("Generated:")
print("  similarity_results.csv")
print("  similarity_summary.csv")
print("  diversity_ranking.csv")
print("  unique_identities.png")
print("  mean_similarity.png")
print("  mean_nearest_neighbor.png")
print("  largest_cluster.png")
print("  pairs_080.png")
print("  pairs_085.png")
print("  pairs_090.png")
print("  diversity_score.png")