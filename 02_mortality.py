import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, roc_auc_score, roc_curve)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import resample
import joblib
import os

os.makedirs("models", exist_ok=True)
os.makedirs("plots",  exist_ok=True)

# ── Load Data ─────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv("processed_data.csv")

# ── Extra Feature Engineering ─────────────────────────────────────────
df["is_emergency"]     = (df["admission_type"] == "EMERGENCY").astype(int)
df["high_abnormal"]    = (df["abnormal_pct"] > 50).astype(int)
df["long_stay"]        = (df["los_days"] > 7).astype(int)
df["lab_burden"]       = df["total_labs"] * df["abnormal_pct"] / 100

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
    "admission_type",
    "insurance",
    "gender"
]

target = "hospital_expire_flag"
df_model = df[features + [target]].copy()

# ── Encode Categorical Columns ────────────────────────────────────────
le = LabelEncoder()
for col in ["admission_type", "insurance", "gender"]:
    df_model[col] = le.fit_transform(df_model[col].astype(str))

# ── Handle Class Imbalance using Oversampling ─────────────────────────
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
print("Survived:", len(df_balanced[df_balanced[target]==0]))
print("Died    :", len(df_balanced[df_balanced[target]==1]))

# ── Split Data ────────────────────────────────────────────────────────
X = df_balanced[features]
y = df_balanced[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining samples :", len(X_train))
print("Testing  samples :", len(X_test))

# ── Train 3 Models and Compare ────────────────────────────────────────
models = {
    "Random Forest"       : RandomForestClassifier(
                                n_estimators=200,
                                max_depth=8,
                                random_state=42
                            ),
    "Gradient Boosting"   : GradientBoostingClassifier(
                                n_estimators=200,
                                max_depth=4,
                                random_state=42
                            ),
    "Logistic Regression" : LogisticRegression(
                                max_iter=1000,
                                random_state=42
                            )
}

print("\n========== MODEL COMPARISON ==========")
best_model     = None
best_accuracy  = 0
best_name      = ""
results        = {}

for name, m in models.items():
    m.fit(X_train, y_train)
    y_pred     = m.predict(X_test)
    y_prob     = m.predict_proba(X_test)[:, 1]
    acc        = accuracy_score(y_test, y_pred)
    auc        = roc_auc_score(y_test, y_prob)
    cv_scores  = cross_val_score(m, X, y, cv=5, scoring="accuracy")

    results[name] = {
        "accuracy"   : acc,
        "roc_auc"    : auc,
        "cv_mean"    : cv_scores.mean(),
        "cv_std"     : cv_scores.std()
    }

    print(f"\n{name}")
    print(f"  Accuracy       : {acc*100:.2f}%")
    print(f"  ROC-AUC        : {auc:.4f}")
    print(f"  CV Accuracy    : {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")

    if acc > best_accuracy:
        best_accuracy = acc
        best_model    = m
        best_name     = name

print(f"\nBest Model: {best_name} with {best_accuracy*100:.2f}% accuracy")

# ── Full Report for Best Model ────────────────────────────────────────
y_pred_best = best_model.predict(X_test)
y_prob_best = best_model.predict_proba(X_test)[:, 1]

print("\n========== BEST MODEL REPORT ==========")
print(classification_report(y_test, y_pred_best,
      target_names=["Survived","Died"]))

# ── Plot 1: Model Comparison ──────────────────────────────────────────
plt.figure(figsize=(8,5))
model_names = list(results.keys())
accuracies  = [results[m]["accuracy"]*100 for m in model_names]
aucs        = [results[m]["roc_auc"]*100  for m in model_names]
x = np.arange(len(model_names))
width = 0.35
plt.bar(x - width/2, accuracies, width, label="Accuracy %",  color="#3498db", edgecolor="black")
plt.bar(x + width/2, aucs,       width, label="ROC-AUC %",   color="#e74c3c", edgecolor="black")
plt.xticks(x, model_names, rotation=10)
plt.ylabel("Score %")
plt.title("Model Comparison - Mortality Prediction")
plt.legend()
plt.tight_layout()
plt.savefig("plots/07_model_comparison.png")
plt.close()
print("\nPlot 1 saved - Model Comparison!")

# ── Plot 2: Confusion Matrix ──────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(6,4))
plt.imshow(cm, cmap="Blues")
plt.title(f"Confusion Matrix - {best_name}")
plt.colorbar()
plt.xticks([0,1], ["Survived","Died"])
plt.yticks([0,1], ["Survived","Died"])
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i,j]),
                 ha="center", va="center",
                 fontsize=16, fontweight="bold",
                 color="white" if cm[i,j] > cm.max()/2 else "black")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("plots/08_confusion_matrix.png")
plt.close()
print("Plot 2 saved - Confusion Matrix!")

# ── Plot 3: ROC Curve ─────────────────────────────────────────────────
plt.figure(figsize=(6,5))
colors = ["#e74c3c", "#3498db", "#2ecc71"]
for (name, m), color in zip(models.items(), colors):
    y_prob = m.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, color=color, linewidth=2,
             label=f"{name} (AUC={auc:.3f})")
plt.plot([0,1],[0,1], color="gray", linestyle="--")
plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.savefig("plots/09_roc_curve.png")
plt.close()
print("Plot 3 saved - ROC Curve!")

# ── Plot 4: Feature Importance ────────────────────────────────────────
if hasattr(best_model, "feature_importances_"):
    plt.figure(figsize=(8,5))
    importances = best_model.feature_importances_
    indices     = np.argsort(importances)
    plt.barh(
        [features[i] for i in indices],
        importances[indices],
        color="#9b59b6", edgecolor="black"
    )
    plt.title(f"Feature Importance - {best_name}")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig("plots/10_feature_importance.png")
    plt.close()
    print("Plot 4 saved - Feature Importance!")

# ── Save Best Model ───────────────────────────────────────────────────
joblib.dump(best_model, "models/mortality_model.pkl")
print(f"\nBest model ({best_name}) saved!")
print("\nMortality Prediction COMPLETE!")