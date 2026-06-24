"""
Kaggle House Prices — Machine Learning Project
===============================================
Dataset: https://www.kaggle.com/c/house-prices-advanced-regression-techniques
File needed: train.csv (place in the same folder as this script)

Concepts covered:
  - Handling missing values (two strategies: drop columns, fill values)
  - Encoding categorical columns (text → numbers)
  - Comparing two models: Linear Regression vs Random Forest
  - Feature importance
  - Evaluating with MAE and R²
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# ── STEP 1: Load the real Kaggle dataset ──────────────────────────────────────
df = pd.read_csv("train.csv")
print(f"Loaded: {df.shape[0]} houses, {df.shape[1]} columns")

# ── STEP 2: Drop columns with too many missing values ─────────────────────────
# If a column is missing more than 50% of its values, it's not very useful.
# thresh = minimum number of NON-null values needed to keep the column.
thresh = len(df) * 0.5
df = df.dropna(axis=1, thresh=thresh)
print(f"After dropping high-missing columns: {df.shape[1]} columns remain")

# ── STEP 3: Separate features (X) and target (y) ─────────────────────────────
df = df.drop(columns=["Id"])      # Id is just a row number, not useful
X = df.drop(columns=["SalePrice"])
y = df["SalePrice"]

# ── STEP 4: Handle remaining missing values ───────────────────────────────────
# Numeric columns → fill missing with the median of that column
# (median is better than mean when there are outliers)
num_cols = X.select_dtypes(include='number').columns
X[num_cols] = X[num_cols].fillna(X[num_cols].median())

# Categorical (text) columns → fill missing with the most common value
cat_cols = X.select_dtypes(include=['object', 'str']).columns
X[cat_cols] = X[cat_cols].fillna(X[cat_cols].mode().iloc[0])

print(f"Missing values remaining: {X.isnull().sum().sum()}")  # should be 0

# ── STEP 5: Encode categorical columns ────────────────────────────────────────
# ML models only understand numbers. LabelEncoder converts each unique
# text value to an integer: e.g. "Grvl"→0, "Pave"→1, "NA"→2
le = LabelEncoder()
for col in cat_cols:
    X[col] = le.fit_transform(X[col].astype(str))

# ── STEP 6: Train / Test split ────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTraining: {len(X_train)} houses | Test: {len(X_test)} houses")

# ── STEP 7: Train Model 1 — Linear Regression ─────────────────────────────────
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr  = r2_score(y_test, y_pred_lr)

# ── STEP 8: Train Model 2 — Random Forest ─────────────────────────────────────
# Random Forest builds many decision trees and averages them.
# n_estimators = how many trees (more = better but slower)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf  = r2_score(y_test, y_pred_rf)

# ── STEP 9: Compare results ────────────────────────────────────────────────────
print(f"\n=== Model Comparison ===")
print(f"{'Model':<22} {'MAE':>12} {'R²':>8}")
print(f"{'Linear Regression':<22} ${mae_lr:>11,.0f} {r2_lr:>8.3f}")
print(f"{'Random Forest':<22} ${mae_rf:>11,.0f} {r2_rf:>8.3f}")
# MAE = on average, how many $ off our predictions are
# R²  = 1.0 is perfect; closer to 1.0 is better

# ── STEP 10: Which features matter most? (Random Forest) ──────────────────────
feat_imp = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
}).sort_values("Importance", ascending=False).head(10)

print("\n=== Top 10 Most Important Features ===")
print(feat_imp.to_string(index=False))

# ── STEP 11: Visualise ────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Kaggle House Prices — Model Results", fontsize=14, fontweight="bold")

# Chart 1: Linear Regression — Actual vs Predicted
axes[0,0].scatter(y_test, y_pred_lr, alpha=0.4, color="steelblue", edgecolors="white", linewidths=0.3)
axes[0,0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=1.5)
axes[0,0].set_title(f"Linear Regression  (R²={r2_lr:.3f})")
axes[0,0].set_xlabel("Actual Price ($)")
axes[0,0].set_ylabel("Predicted Price ($)")

# Chart 2: Random Forest — Actual vs Predicted
axes[0,1].scatter(y_test, y_pred_rf, alpha=0.4, color="seagreen", edgecolors="white", linewidths=0.3)
axes[0,1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=1.5)
axes[0,1].set_title(f"Random Forest  (R²={r2_rf:.3f})")
axes[0,1].set_xlabel("Actual Price ($)")
axes[0,1].set_ylabel("Predicted Price ($)")

# Chart 3: Feature Importances
axes[1,0].barh(feat_imp["Feature"][::-1], feat_imp["Importance"][::-1], color="steelblue")
axes[1,0].set_title("Top 10 Most Important Features")
axes[1,0].set_xlabel("Importance Score")

# Chart 4: Price Distribution
axes[1,1].hist(y, bins=40, color="steelblue", edgecolor="white")
axes[1,1].set_title("Sale Price Distribution")
axes[1,1].set_xlabel("Price ($)")
axes[1,1].set_ylabel("Count")

plt.tight_layout()
plt.savefig("kaggle_results.png", dpi=130, bbox_inches="tight")
plt.show()
print("\nPlot saved as kaggle_results.png")
