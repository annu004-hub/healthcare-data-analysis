import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import joblib
import os

os.makedirs("models", exist_ok=True)
os.makedirs("plots",  exist_ok=True)

# ── Load Data ─────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv("processed_data.csv")

# ── Feature Engineering ───────────────────────────────────────────────
df["is_emergency"]  = (df["admission_type"] == "EMERGENCY").astype(int)
df["high_abnormal"] = (df["abnormal_pct"] > 50).astype(int)
df["lab_burden"]    = df["total_labs"] * df["abnormal_pct"] / 100
df["is_male"]       = (df["gender"] == "M").astype(int)

features = [
    "age",
    "los_days",
    "total_labs",
    "abnormal_pct",
    "unique_tests",
    "lab_burden",
    "is_emergency",
    "high_abnormal",
    "hospital_expire_flag"
]

df_model = df[features].copy().fillna(0)

# ── Scale Features ────────────────────────────────────────────────────
print("Scaling features...")
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(df_model)

# ── Find Best K using Elbow Method ────────────────────────────────────
print("Finding optimal number of clusters...")
inertias        = []
silhouette_scores = []
k_range         = range(2, 8)

for k in k_range:
    km  = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, km.labels_))
    print(f"  K={k} | Inertia={km.inertia_:.2f} | Silhouette={silhouette_score(X_scaled, km.labels_):.4f}")

# ── Plot 1: Elbow Curve ───────────────────────────────────────────────
plt.figure(figsize=(8,4))
plt.plot(list(k_range), inertias, marker="o", color="#e74c3c", linewidth=2)
plt.title("Elbow Method - Optimal Number of Clusters")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.xticks(list(k_range))
plt.tight_layout()
plt.savefig("plots/19_elbow_curve.png")
plt.close()
print("\nPlot 1 saved - Elbow Curve!")

# ── Plot 2: Silhouette Scores ─────────────────────────────────────────
plt.figure(figsize=(8,4))
plt.plot(list(k_range), silhouette_scores, marker="s",
         color="#3498db", linewidth=2)
plt.title("Silhouette Score by Number of Clusters")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.xticks(list(k_range))
plt.tight_layout()
plt.savefig("plots/20_silhouette_scores.png")
plt.close()
print("Plot 2 saved - Silhouette Scores!")

# ── Train Final KMeans with K=4 ───────────────────────────────────────
best_k = silhouette_scores.index(max(silhouette_scores)) + 2
print(f"\nBest K = {best_k} (highest silhouette score)")

kmeans   = KMeans(n_clusters=best_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
df["cluster"] = clusters

# ── Cluster Profiles ──────────────────────────────────────────────────
print("\n========== CLUSTER PROFILES ==========")
cluster_summary = df.groupby("cluster").agg(
    patients        = ("subject_id",          "count"),
    avg_age         = ("age",                 "mean"),
    avg_los         = ("los_days",            "mean"),
    avg_abnormal    = ("abnormal_pct",        "mean"),
    avg_total_labs  = ("total_labs",          "mean"),
    mortality_rate  = ("hospital_expire_flag","mean")
).round(2)

cluster_summary["mortality_rate"] = (
    cluster_summary["mortality_rate"] * 100
).round(2)

print(cluster_summary.to_string())

# ── Assign Severity Labels ────────────────────────────────────────────
cluster_summary["severity"] = ""
for idx in cluster_summary.index:
    score = (
        cluster_summary.loc[idx, "avg_los"] / 10 +
        cluster_summary.loc[idx, "avg_abnormal"] / 100 +
        cluster_summary.loc[idx, "mortality_rate"] / 100
    )
    cluster_summary.loc[idx, "severity_score"] = round(score, 4)

severity_order = cluster_summary["severity_score"].rank(ascending=False)
severity_labels = {1: "CRITICAL", 2: "HIGH", 3: "MODERATE", 4: "LOW"}

for idx in cluster_summary.index:
    rank = int(severity_order[idx])
    cluster_summary.loc[idx, "severity"] = severity_labels.get(rank, "LOW")

df["severity"] = df["cluster"].map(
    cluster_summary["severity"].to_dict()
)

print("\n========== SEVERITY DISTRIBUTION ==========")
print(df["severity"].value_counts())

# ── Plot 3: PCA Cluster Visualization ────────────────────────────────
pca      = PCA(n_components=2, random_state=42)
X_pca    = pca.fit_transform(X_scaled)
colors   = ["#e74c3c","#3498db","#2ecc71","#f39c12","#9b59b6","#1abc9c"]

plt.figure(figsize=(8,6))
for i in range(best_k):
    mask = clusters == i
    sev  = cluster_summary.loc[i, "severity"]
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                c=colors[i], label=f"Cluster {i} ({sev})",
                alpha=0.7, edgecolors="black", s=80)
plt.title("Patient Clusters - PCA Visualization")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.tight_layout()
plt.savefig("plots/21_cluster_pca.png")
plt.close()
print("\nPlot 3 saved - PCA Clusters!")

# ── Plot 4: Cluster Profiles Bar Chart ───────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14,5))

axes[0].bar(cluster_summary.index.astype(str),
            cluster_summary["avg_los"],
            color=colors[:best_k], edgecolor="black")
axes[0].set_title("Avg Length of Stay by Cluster")
axes[0].set_xlabel("Cluster")
axes[0].set_ylabel("Days")

axes[1].bar(cluster_summary.index.astype(str),
            cluster_summary["avg_abnormal"],
            color=colors[:best_k], edgecolor="black")
axes[1].set_title("Avg Abnormal Lab % by Cluster")
axes[1].set_xlabel("Cluster")
axes[1].set_ylabel("Abnormal %")

axes[2].bar(cluster_summary.index.astype(str),
            cluster_summary["mortality_rate"],
            color=colors[:best_k], edgecolor="black")
axes[2].set_title("Mortality Rate % by Cluster")
axes[2].set_xlabel("Cluster")
axes[2].set_ylabel("Mortality %")

plt.suptitle("Cluster Profiles Overview", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/22_cluster_profiles.png")
plt.close()
print("Plot 4 saved - Cluster Profiles!")

# ── Plot 5: Severity Distribution ────────────────────────────────────
plt.figure(figsize=(7,4))
sev_counts = df["severity"].value_counts()
sev_colors = {"CRITICAL":"#e74c3c","HIGH":"#e67e22",
              "MODERATE":"#f1c40f","LOW":"#2ecc71"}
bar_colors = [sev_colors.get(s,"#3498db") for s in sev_counts.index]
plt.bar(sev_counts.index, sev_counts.values,
        color=bar_colors, edgecolor="black")
plt.title("Patient Severity Distribution")
plt.ylabel("Number of Patients")
for i, v in enumerate(sev_counts.values):
    plt.text(i, v + 0.5, str(v), ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig("plots/23_severity_distribution.png")
plt.close()
print("Plot 5 saved - Severity Distribution!")

# ── Save Model & Results ──────────────────────────────────────────────
joblib.dump(kmeans, "models/clustering_model.pkl")
joblib.dump(scaler, "models/clustering_scaler.pkl")
df.to_csv("final_data_with_clusters.csv", index=False)

print("\n========== FINAL SUMMARY ==========")
print(cluster_summary[["patients","avg_age","avg_los",
                        "avg_abnormal","mortality_rate","severity"]].to_string())
print("\nClustering model saved!")
print("final_data_with_clusters.csv saved!")
print("\nPatient Severity Clustering COMPLETE!")