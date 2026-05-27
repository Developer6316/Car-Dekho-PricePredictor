"""
Car Price Prediction - SUPER FAST for Colab/Local
No GridSearch, uses optimized Random Forest with sampling
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import json
import time

start_time = time.time()

print("="*60)
print("CAR PRICE PREDICTION - FAST TRAINING")
print("="*60)

# 1. Load data from GitHub (same as yours)
url = "https://raw.githubusercontent.com/manishkr1754/CarDekho_Used_Car_Price_Prediction/main/notebooks/data/cardekho_dataset.csv"
df = pd.read_csv(url)
print(f"Loaded shape: {df.shape}")

# 2. Target column
target = 'selling_price'  # from your output

# 3. Drop 'Unnamed: 0' if exists
if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)

# 4. Drop 'car_name' (already done in your script, but ensure)
if 'car_name' in df.columns:
    df = df.drop('car_name', axis=1)

# 5. Handle missing values (simplest)
df = df.dropna(subset=[target])
for col in df.select_dtypes(include=['object']).columns:
    df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
for col in df.select_dtypes(include=['int64','float64']).columns:
    df[col].fillna(df[col].median(), inplace=True)

# 6. Split data BEFORE expensive encoding (smaller training set speeds up)
X = df.drop(target, axis=1)
y = df[target]

# Option: sample 50% of data for even faster training (uncomment if still slow)
# X, _, y, _ = train_test_split(X, y, train_size=0.5, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# 7. Identify columns
numeric_cols = X_train.select_dtypes(include=['int64','float64']).columns.tolist()
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
print(f"Numeric: {len(numeric_cols)}, Categorical: {len(categorical_cols)}")

# For high-cardinality columns like 'brand' and 'model', we'll keep them but one-hot will expand.
# To speed up, we can limit categories to top 10 most frequent for 'model' (optional)
if 'model' in categorical_cols:
    top_models = X_train['model'].value_counts().head(10).index
    X_train['model'] = X_train['model'].apply(lambda x: x if x in top_models else 'Other')
    X_test['model'] = X_test['model'].apply(lambda x: x if x in top_models else 'Other')
    print("Limited 'model' to top 10 categories + Other")

# 8. Preprocessor with handle_unknown='ignore'
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
])

# 9. Random Forest with reasonable defaults (no grid search)
print("\nTraining Random Forest (default params, no tuning)...")
rf = RandomForestRegressor(
    n_estimators=100,      # 100 trees is enough
    max_depth=20,          # limit depth to speed up
    min_samples_split=10,  # higher value = faster
    n_jobs=-1,             # use all CPU cores
    random_state=42
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', rf)
])

# Fit (this should take 1-2 minutes on Colab CPU)
pipeline.fit(X_train, y_train)

# 10. Evaluate
y_pred = pipeline.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n✅ Performance:")
print(f"   R²: {r2:.4f}")
print(f"   MAE: {mae:.2f}")
print(f"   RMSE: {rmse:.2f}")

# 11. Save model and plots in CURRENT folder
current_dir = "."  # This is the folder where the script runs (on Colab, it's /content)
model_path = f"{current_dir}/best_car_price_model.pkl"
joblib.dump(pipeline, model_path)
print(f"✅ Model saved: {model_path}")

# Save metrics
metrics = {'RandomForest': {'R²': r2, 'MAE': mae, 'RMSE': rmse}}
with open(f"{current_dir}/model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# 12. Plots
fig, axes = plt.subplots(1, 2, figsize=(12,5))
axes[0].scatter(y_test, y_pred, alpha=0.5)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axes[0].set_xlabel('Actual Price')
axes[0].set_ylabel('Predicted Price')
axes[0].set_title(f'Random Forest (R²={r2:.3f})')

residuals = y_test - y_pred
axes[1].scatter(y_pred, residuals, alpha=0.5)
axes[1].axhline(y=0, color='r', linestyle='--')
axes[1].set_xlabel('Predicted Price')
axes[1].set_ylabel('Residuals')
axes[1].set_title('Residual Plot')

plt.tight_layout()
plt.savefig(f"{current_dir}/car_price_results.png", dpi=100)
plt.close()
print("✅ Plot saved: car_price_results.png")

# Feature importance (optional)
# Get feature names after one-hot encoding
preprocessor.fit(X_train)
cat_features = []
if categorical_cols:
    # Get one-hot column names
    ohe = preprocessor.named_transformers_['cat']
    cat_features = ohe.get_feature_names_out(categorical_cols).tolist()
all_features = numeric_cols + cat_features
importances = rf.feature_importances_
if len(importances) == len(all_features):
    feat_imp = pd.Series(importances, index=all_features).sort_values(ascending=False).head(10)
    plt.figure(figsize=(10,6))
    feat_imp.plot(kind='barh')
    plt.title('Top 10 Feature Importances')
    plt.tight_layout()
    plt.savefig(f"{current_dir}/feature_importance.png")
    plt.close()
    print("✅ Feature importance saved: feature_importance.png")

elapsed = time.time() - start_time
print(f"\n⏱️ Total time: {elapsed:.2f} seconds")
print("="*60)
print("Training complete. The .pkl file is saved and can be used")
