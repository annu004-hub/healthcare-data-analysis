import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from db_connect import run_query
import os

os.makedirs("plots", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ── Pull data from MySQL ──────────────────────────────────────────────
print("Fetching data from MySQL...")
admissions = run_query("SELECT * FROM admissions")
patients   = run_query("SELECT * FROM patients")
labevents  = run_query("SELECT * FROM labevents")
print("Data fetched!")
print("Admissions :", len(admissions))
print("Patients   :", len(patients))
print("Lab Events :", len(labevents))

# ── Merge ─────────────────────────────────────────────────────────────
df = admissions.merge(patients, on="subject_id", how="left")

# ── Feature Engineering ───────────────────────────────────────────────
df["admittime"] = pd.to_datetime(df["admittime"])
df["dischtime"] = pd.to_datetime(df["dischtime"])
df["dob"]       = pd.to_datetime(df["dob"])

df["los_days"] = (df["dischtime"] - df["admittime"]).dt.days
df["age"]      = ((df["admittime"] - df["dob"]).dt.days / 365.25).astype(int)

# ── Lab Summary per Patient ───────────────────────────────────────────
lab_features = labevents.groupby("subject_id").agg(
    total_labs    = ("row_id", "count"),
    abnormal_labs = ("flag",   lambda x: (x == "abnormal").sum()),
    unique_tests  = ("itemid", "nunique")
).reset_index()

lab_features["abnormal_pct"] = np.round(
    100.0 * lab_features["abnormal_labs"] / lab_features["total_labs"], 2
)

df = df.merge(lab_features, on="subject_id", how="left")

# ── Fix fillna for datetime columns ───────────────────────────────────
numeric_cols = df.select_dtypes(include=["number"]).columns
df[numeric_cols] = df[numeric_cols].fillna(0)
df["los_days"]   = df["los_days"].fillna(0)
df["age"]        = df["age"].fillna(0)

print("\nFinal Dataset Shape:", df.shape)
print(df[["age","los_days","abnormal_pct","hospital_expire_flag"]].describe())

# ── Plot 1: Mortality Count ───────────────────────────────────────────
plt.figure(figsize=(6,4))
labels = ["Survived", "Died"]
counts = df["hospital_expire_flag"].value_counts().sort_index()
plt.bar(labels, counts, color=["#2ecc71","#e74c3c"], edgecolor="black")
plt.title("Overall Mortality Count")
plt.ylabel("Number of Patients")
for i, v in enumerate(counts):
    plt.text(i, v + 1, str(v), ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig("plots/01_mortality_count.png")
plt.close()
print("Plot 1 saved!")

# ── Plot 2: LOS Distribution ──────────────────────────────────────────
plt.figure(figsize=(7,4))
plt.hist(df["los_days"].clip(0,40), bins=20,
         color="#3498db", edgecolor="black")
plt.title("Length of Stay Distribution")
plt.xlabel("Days")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("plots/02_los_distribution.png")
plt.close()
print("Plot 2 saved!")

# ── Plot 3: Age Distribution ──────────────────────────────────────────
plt.figure(figsize=(7,4))
plt.hist(df["age"].clip(0,100), bins=20,
         color="#9b59b6", edgecolor="black")
plt.title("Patient Age Distribution")
plt.xlabel("Age (years)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("plots/03_age_distribution.png")
plt.close()
print("Plot 3 saved!")

# ── Plot 4: Admission Type ────────────────────────────────────────────
plt.figure(figsize=(6,4))
adm_counts = df["admission_type"].value_counts()
plt.bar(adm_counts.index, adm_counts.values,
        color=["#e74c3c","#3498db","#f39c12"], edgecolor="black")
plt.title("Admissions by Type")
plt.ylabel("Count")
for i, v in enumerate(adm_counts.values):
    plt.text(i, v + 1, str(v), ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig("plots/04_admission_type.png")
plt.close()
print("Plot 4 saved!")

# ── Plot 5: Abnormal Lab % by Mortality ───────────────────────────────
plt.figure(figsize=(6,4))
survived = df[df["hospital_expire_flag"]==0]["abnormal_pct"]
died     = df[df["hospital_expire_flag"]==1]["abnormal_pct"]
plt.boxplot([survived, died],
            labels=["Survived","Died"],
            patch_artist=True,
            boxprops=dict(facecolor="#3498db"),
            medianprops=dict(color="red", linewidth=2))
plt.title("Abnormal Lab % vs Mortality")
plt.ylabel("Abnormal Lab %")
plt.tight_layout()
plt.savefig("plots/05_abnormal_vs_mortality.png")
plt.close()
print("Plot 5 saved!")

# ── Plot 6: Insurance vs Mortality ───────────────────────────────────
plt.figure(figsize=(7,4))
ins = df.groupby("insurance")["hospital_expire_flag"].mean() * 100
ins = ins.sort_values()
plt.barh(ins.index, ins.values, color="#e67e22", edgecolor="black")
plt.title("Mortality Rate by Insurance (%)")
plt.xlabel("Mortality %")
plt.tight_layout()
plt.savefig("plots/06_insurance_mortality.png")
plt.close()
print("Plot 6 saved!")

# ── Save processed data ───────────────────────────────────────────────
df.to_csv("processed_data.csv", index=False)
print("\nAll done! processed_data.csv saved - Ready for ML models!")
