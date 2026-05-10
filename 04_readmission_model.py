import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, roc_auc_score, roc_curve)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
import joblib
import os

os.makedirs("models", exist_ok=True)
os.makedirs("plots",  exist_ok=True)

# ── Load Data ─────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv("processed_data.csv")

# ── Create Readmission Label ──────────────────────────────────────────
print("Creating readmission labels...")
df["admittime"] = pd.to_datetime(df["admittime"])
df["dischtime"] = pd.to_datetime(df["dischtime"])
df = df.sort_values(["subject_id","admittime"]).reset_index(drop=True)

df["prev_dischtime"] = df.groupby("subject_id")["dischtime"].shift(1)
df["days_since_discharge"] = (
    df["admittime"] - df["prev_dischtime"]
).dt.days

df["readmitted_30d"] = (
    (df["days_since_discharge"] <= 30) &
    (df["days_since_discharge"] >= 0)
).astype(int)

print("Readmission distribution:")
print(df["readmitted_30d"].value_counts())

# ── Feature Engineering ───────────────────────────────────────────────
df["is_emergency"]  = (df["admission_type"] == "EMERGENCY").astype(int)
df["high_abnormal"] = (df["abnormal_pct"] > 50).astype(int)
df["long_stay"]     = (df["los_days"] > 7).astype(int)
df["lab_burden"]    = df["total_labs"] * df["abnormal_pct"] / 100
df["is_male"]       = (df["gender"] == "M").astype(int)

features = [
    "age",
    "los_days",
    "total_labs",
    "abnormal_labs",
    "abnormal_pct",
    "unique_tests",
    "is_emergency",
    "high_abnormal",
    "long_stay",
    "lab_burden",
    "is_male",
    "hospital_expire_flag",
    "admission_type",
    "insurance"
]

target = "readmitted_30d"

df_model = df[features + [target]].copy()
df_model = df_model.dropna()

# ── Encode Categorical ────────────────────────────────────────────────
le = LabelEncoder()
for col in ["admission_type","insurance"]:
    df_model[col] = le.fit_transform(df_model[col].astype(str))

# ── Handle Class Imbalance ────────────────────────────────────────────
print("\nHandling class imbalance...")
df_majority = df_model[df_model[target] == 0]
df_minority = df_model[df_model[target] == 1]

df_minority_upsampled = resample(
    df_minority,
    replace=True,
    n_samples=len(df_majority),
    random_state=42
)

df_balanced = pd.concat([df_majority, df_minority_upsampled])
print("Balanced dataset size:", len(df_balanced))

# ── Split Data ────────────────────────────────────────────────────────
X = df_balanced[features]
y = df_balanced[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training samples :", len(X_train))
print("Testing  samples :", len(X_test))

# ── Train 3 Models ────────────────────────────────────────────────────
models = {
    "Logistic Regression" : LogisticRegression(
                                max_iter=1000,
                                random_state=42
                            ),
    "Random Forest"       : RandomForestClassifier(
                                n_estimators=200,
                                max_depth=8,
                                random_state=42
                            ),
    "Gradient Boosting"   : GradientBoostingClassifier(
                                n_estimators=200,
                                max_depth=4,
                                random_state=42
                            )
}

print("\n========== MODEL COMPARISON ==========")
best_model    = None
best_accuracy = 0
best_name     = ""
results       = {}

for name, m in models.items():
    m.fit(X_train, y_train)
    y_pred = m.predict(X_test)
    y_prob = m.predict_proba(X_test)[:, 1]
    acc    = accuracy_score(y_test, y_pred)
    auc    = roc_auc_score(y_test, y_prob)
    cv     = cross_val_score(m, X, y, cv=5, scoring="accuracy")

    results[name] = {"accuracy": acc, "roc_auc": auc, "cv_mean": cv.mean()}

    print(f"\n{name}")
    print(f"  Accuracy    : {acc*100:.2f}%")
    print(f"  ROC-AUC     : {auc:.4f}")
    print(f"  CV Accuracy : {cv.mean()*100:.2f}% (+/- {cv.std()*100:.2f}%)")

    if acc > best_accuracy:
        best_accuracy = acc
        best_model    = m
        best_name     = name

print(f"\nBest Model: {best_name} with {best_accuracy*100:.2f}% accuracy")

# ── Full Report ───────────────────────────────────────────────────────
y_pred_best = best_model.predict(X_test)
y_prob_best = best_model.predict_proba(X_test)[:, 1]
print("\n========== BEST MODEL REPORT ==========")
print(classification_report(y_test, y_pred_best,
      target_names=["Not Readmitted","Readmitted"]))

# ── Plot 1: Model Comparison ──────────────────────────────────────────
plt.figure(figsize=(8,5))
model_names = list(results.keys())
accuracies  = [results[m]["accuracy"]*100 for m in model_names]
aucs        = [results[m]["roc_auc"]*100  for m in model_names]
x     = np.arange(len(model_names))
width = 0.35
plt.bar(x - width/2, accuracies, width, label="Accuracy %",
        color="#3498db", edgecolor="black")
plt.bar(x + width/2, aucs,       width, label="ROC-AUC %",
        color="#e74c3c", edgecolor="black")
plt.xticks(x, model_names, rotation=10)
plt.ylabel("Score %")
plt.title("Model Comparison - 30-Day Readmission")
plt.legend()
plt.tight_layout()
plt.savefig("plots/15_readmission_model_comparison.png")
plt.close()
print("\nPlot 1 saved - Model Comparison!")

# ── Plot 2: Confusion Matrix ──────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(6,4))
plt.imshow(cm, cmap="Greens")
plt.title(f"Confusion Matrix - {best_name}")
plt.colorbar()
plt.xticks([0,1], ["Not Readmitted","Readmitted"])
plt.yticks([0,1], ["Not Readmitted","Readmitted"])
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i,j]),
                 ha="center", va="center",
                 fontsize=16, fontweight="bold",
                 color="white" if cm[i,j] > cm.max()/2 else "black")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("plots/16_readmission_confusion_matrix.png")
plt.close()
print("Plot 2 saved - Confusion Matrix!")

# ── Plot 3: ROC Curve ─────────────────────────────────────────────────
plt.figure(figsize=(6,5))
colors = ["#2ecc71","#3498db","#e74c3c"]
for (name, m), color in zip(models.items(), colors):
    y_prob   = m.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc      = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, color=color, linewidth=2,
             label=f"{name} (AUC={auc:.3f})")
plt.plot([0,1],[0,1], color="gray", linestyle="--")
plt.title("ROC Curve - 30-Day Readmission")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.savefig("plots/17_readmission_roc.png")
plt.close()
print("Plot 3 saved - ROC Curve!")

# ── Plot 4: Readmission by Diagnosis ─────────────────────────────────
plt.figure(figsize=(9,5))
diag = df.groupby("diagnosis")["readmitted_30d"].mean() * 100
diag = diag.sort_values(ascending=False).head(10)
plt.barh(diag.index, diag.values, color="#e74c3c", edgecolor="black")
plt.title("Top 10 Diagnoses by Readmission Rate (%)")
plt.xlabel("Readmission Rate %")
plt.tight_layout()
plt.savefig("plots/18_readmission_by_diagnosis.png")
plt.close()
print("Plot 4 saved - Readmission by Diagnosis!")

# ── Save Best Model ───────────────────────────────────────────────────
joblib.dump(best_model, "models/readmission_model.pkl")
print(f"\nBest model ({best_name}) saved!")
print("\n30-Day Readmission Prediction COMPLETE!")