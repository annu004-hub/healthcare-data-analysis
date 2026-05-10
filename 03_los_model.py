import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score)
from sklearn.preprocessing import LabelEncoder
import joblib
import os

os.makedirs("models", exist_ok=True)
os.makedirs("plots",  exist_ok=True)

# ── Load Data ─────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv("processed_data.csv")

# ── Extra Feature Engineering ─────────────────────────────────────────
df["is_emergency"]  = (df["admission_type"] == "EMERGENCY").astype(int)
df["high_abnormal"] = (df["abnormal_pct"] > 50).astype(int)
df["lab_burden"]    = df["total_labs"] * df["abnormal_pct"] / 100
df["is_male"]       = (df["gender"] == "M").astype(int)

features = [
    "age",
    "total_labs",
    "abnormal_labs",
    "abnormal_pct",
    "unique_tests",
    "is_emergency",
    "high_abnormal",
    "lab_burden",
    "is_male",
    "hospital_expire_flag",
    "admission_type",
    "insurance"
]

target = "los_days"

df_model = df[features + [target]].copy()
df_model = df_model[df_model[target] > 0]  # remove 0 day stays
df_model = df_model[df_model[target] < 60] # remove extreme outliers

# ── Encode Categorical Columns ────────────────────────────────────────
le = LabelEncoder()
for col in ["admission_type", "insurance"]:
    df_model[col] = le.fit_transform(df_model[col].astype(str))

# ── Split Data ────────────────────────────────────────────────────────
X = df_model[features]
y = df_model[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training samples :", len(X_train))
print("Testing  samples :", len(X_test))

# ── Train 3 Models ────────────────────────────────────────────────────
models = {
    "Linear Regression"   : LinearRegression(),
    "Random Forest"       : RandomForestRegressor(
                                n_estimators=200,
                                max_depth=8,
                                random_state=42
                            ),
    "Gradient Boosting"   : GradientBoostingRegressor(
                                n_estimators=200,
                                max_depth=4,
                                random_state=42
                            )
}

print("\n========== MODEL COMPARISON ==========")
best_model    = None
best_r2       = -999
best_name     = ""
results       = {}

for name, m in models.items():
    m.fit(X_train, y_train)
    y_pred = m.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    results[name] = {
        "mae"  : mae,
        "rmse" : rmse,
        "r2"   : r2
    }

    print(f"\n{name}")
    print(f"  MAE  (Mean Abs Error) : {mae:.2f} days")
    print(f"  RMSE (Root Mean Sq)   : {rmse:.2f} days")
    print(f"  R2   Score            : {r2:.4f}")

    if r2 > best_r2:
        best_r2    = r2
        best_model = m
        best_name  = name

print(f"\nBest Model: {best_name} with R2 = {best_r2:.4f}")

# ── Plot 1: Model Comparison (R2 Score) ──────────────────────────────
plt.figure(figsize=(8,5))
model_names = list(results.keys())
r2_scores   = [results[m]["r2"] for m in model_names]
colors      = ["#3498db","#e74c3c","#2ecc71"]
bars = plt.bar(model_names, r2_scores, color=colors, edgecolor="black")
plt.title("Model Comparison - R2 Score (LOS Prediction)")
plt.ylabel("R2 Score")
plt.ylim(0, 1)
for bar, val in zip(bars, r2_scores):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.01,
             f"{val:.4f}", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig("plots/11_los_model_comparison.png")
plt.close()
print("\nPlot 1 saved - Model Comparison!")

# ── Plot 2: Actual vs Predicted ───────────────────────────────────────
y_pred_best = best_model.predict(X_test)
plt.figure(figsize=(7,5))
plt.scatter(y_test, y_pred_best, color="#3498db",
            alpha=0.7, edgecolors="black")
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color="red", linestyle="--", linewidth=2)
plt.title(f"Actual vs Predicted LOS - {best_name}")
plt.xlabel("Actual LOS (days)")
plt.ylabel("Predicted LOS (days)")
plt.tight_layout()
plt.savefig("plots/12_actual_vs_predicted.png")
plt.close()
print("Plot 2 saved - Actual vs Predicted!")

# ── Plot 3: Residuals ─────────────────────────────────────────────────
residuals = y_test - y_pred_best
plt.figure(figsize=(7,4))
plt.hist(residuals, bins=20, color="#9b59b6", edgecolor="black")
plt.axvline(x=0, color="red", linestyle="--", linewidth=2)
plt.title(f"Residuals Distribution - {best_name}")
plt.xlabel("Residual (Actual - Predicted)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("plots/13_residuals.png")
plt.close()
print("Plot 3 saved - Residuals!")

# ── Plot 4: Feature Importance ────────────────────────────────────────
if hasattr(best_model, "feature_importances_"):
    plt.figure(figsize=(8,5))
    importances = best_model.feature_importances_
    indices     = np.argsort(importances)
    plt.barh(
        [features[i] for i in indices],
        importances[indices],
        color="#e67e22", edgecolor="black"
    )
    plt.title(f"Feature Importance - {best_name}")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig("plots/14_los_feature_importance.png")
    plt.close()
    print("Plot 4 saved - Feature Importance!")

# ── Save Best Model ───────────────────────────────────────────────────
joblib.dump(best_model, "models/los_model.pkl")
print(f"\nBest model ({best_name}) saved!")
print("\nLength of Stay Prediction COMPLETE!")