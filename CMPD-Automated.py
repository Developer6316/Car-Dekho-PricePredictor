"""
Car Price Prediction - Training & Tuning
Saves model, plots, and metrics in current directory
"""
import warnings
warnings.filterwarnings("ignore")
import sys
if not sys.warnoptions:
    import os
    os.environ["PYTHONWARNINGS"] = "ignore"
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import urllib.request
import os
import matplotlib.pyplot as plt
import joblib
import json

# ============================================
# 1. LOAD DATA
# ============================================
print("="*60)
print("CAR PRICE PREDICTION MODEL - TRAINING")
print("="*60)

print("\n[1] Loading dataset from GitHub...")
url = "https://raw.githubusercontent.com/manishkr1754/CarDekho_Used_Car_Price_Prediction/main/notebooks/data/cardekho_dataset.csv"

try:
    df = pd.read_csv(url)
    print(f"✅ Loaded. Shape: {df.shape}")
except Exception as e:
    print(f"Downloading...")
    os.makedirs("./data/", exist_ok=True)
    urllib.request.urlretrieve(url, "./data/cardekho_dataset.csv")
    df = pd.read_csv("./data/cardekho_dataset.csv")
    print(f"✅ Downloaded and loaded.")

print("\nColumns in dataset:")
print(df.columns.tolist())

# ============================================
# 2. DETECT TARGET COLUMN
# ============================================
print("\n[2] Identifying target column...")
possible_targets = ['selling_price', 'Selling_Price', 'price', 'Price', 'target']
target_col = None
for col in possible_targets:
    if col in df.columns:
        target_col = col
        break

if target_col is None:
    for col in df.columns:
        if 'selling' in col.lower() or 'price' in col.lower():
            target_col = col
            break
    if target_col is None:
        numeric_cols = df.select_dtypes(include=['int64','float64']).columns
        target_col = numeric_cols[-1]
        print(f"⚠️ Using '{target_col}' as target (fallback).")
else:
    print(f"✅ Target column: '{target_col}'")

# ============================================
# 3. DATA CLEANING
# ============================================
print("\n[3] Cleaning data...")
df = df.dropna(subset=[target_col])
print(f"Rows after dropping missing target: {len(df)}")

for col in df.columns:
    if df[col].dtype == 'object':
        df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)

# Feature engineering: Car age
if 'year' in df.columns:
    df['car_age'] = 2025 - df['year']
    df.drop('year', axis=1, inplace=True)
    print("✅ Created 'car_age' from 'year'")
elif 'Year' in df.columns:
    df['car_age'] = 2025 - df['Year']
    df.drop('Year', axis=1, inplace=True)
    print("✅ Created 'car_age' from 'Year'")

# Drop car name columns
for name_col in ['car_name', 'Car_Name']:
    if name_col in df.columns:
        df.drop(name_col, axis=1, inplace=True)
        print(f"✅ Dropped '{name_col}'")

# ============================================
# 4. SEPARATE X AND y
# ============================================
X = df.drop(target_col, axis=1)
y = df[target_col]

numeric_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
print(f"Numeric features: {numeric_cols}")
print(f"Categorical features: {categorical_cols}")

# ============================================
# 5. TRAIN-TEST SPLIT
# ============================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

# ============================================
# 6. PREPROCESSOR
# ============================================
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
])

# ============================================
# 7. HYPERPARAMETER TUNING (GridSearchCV)
# ============================================
print("\n[4] Hyperparameter tuning for Random Forest...")

rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

param_grid = {
    'regressor__n_estimators': [100, 200],
    'regressor__max_depth': [10, 20, None],
    'regressor__min_samples_split': [2, 5]
}

grid_search = GridSearchCV(rf_pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
print(f"Best params: {grid_search.best_params_}")
print(f"Best CV R²: {grid_search.best_score_:.4f}")

# ============================================
# 8. TRAIN AND EVALUATE MULTIPLE MODELS
# ============================================
print("\n[5] Training final models...")

models = {
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.01),
    'RandomForest': best_rf,
    'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = {}
best_model = None
best_r2 = -np.inf
best_name = ""

for name, model in models.items():
    print(f"\n--- {name} ---")
    if name == 'RandomForest':
        pipeline = model
    else:
        pipeline = Pipeline([('preprocessor', preprocessor), ('regressor', model)])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"  Test R²: {r2:.4f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    results[name] = {'R²': r2, 'MAE': mae, 'RMSE': rmse}
    if r2 > best_r2:
        best_r2 = r2
        best_model = pipeline
        best_name = name

# ============================================
# 9. SAVE THE BEST MODEL (current folder)
# ============================================
print("\n[6] Saving best model in current folder...")
current_dir = os.getcwd()  # current directory where script runs
model_path = os.path.join(current_dir, "best_car_price_model.pkl")
joblib.dump(best_model, model_path)
print(f"✅ Model saved to: {model_path}")

metrics_path = os.path.join(current_dir, "model_metrics.json")
with open(metrics_path, 'w') as f:
    json.dump(results, f, indent=4)
print(f"✅ Metrics saved to: {metrics_path}")

# ============================================
# 10. SAVE PLOTS IN CURRENT FOLDER
# ============================================
print("\n[7] Generating and saving plots...")
y_pred_best = best_model.predict(X_test)

fig, axes = plt.subplots(1, 2, figsize=(12,5))
# Actual vs Predicted
axes[0].scatter(y_test, y_pred_best, alpha=0.5)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0].set_xlabel('Actual Price')
axes[0].set_ylabel('Predicted Price')
axes[0].set_title(f'Best: {best_name} (R²={best_r2:.3f})')

# Residuals
residuals = y_test - y_pred_best
axes[1].scatter(y_pred_best, residuals, alpha=0.5)
axes[1].axhline(y=0, color='r', linestyle='--')
axes[1].set_xlabel('Predicted Price')
axes[1].set_ylabel('Residuals')
axes[1].set_title('Residual Plot')

plt.tight_layout()
plot_path = os.path.join(current_dir, "car_price_results.png")
plt.savefig(plot_path, dpi=150)
print(f"✅ Plot saved to: {plot_path}")
plt.close()

# Feature importance (if Random Forest)
if hasattr(best_model.named_steps['regressor'], 'feature_importances_'):
    preprocessor.fit(X_train)
    cat_features = []
    for name, trans, cols in preprocessor.transformers_:
        if name == 'cat':
            if hasattr(trans, 'get_feature_names_out'):
                cat_features = trans.get_feature_names_out(cols).tolist()
            else:
                cat_features = cols
    all_features = numeric_cols + cat_features
    importances = best_model.named_steps['regressor'].feature_importances_
    if len(importances) == len(all_features):
        feat_imp = pd.Series(importances, index=all_features).sort_values(ascending=False).head(10)
        plt.figure(figsize=(10,6))
        feat_imp.plot(kind='barh')
        plt.title('Top 10 Feature Importances')
        plt.tight_layout()
        imp_path = os.path.join(current_dir, "feature_importance.png")
        plt.savefig(imp_path)
        print(f"✅ Feature importance plot saved to: {imp_path}")
        plt.close()

# ============================================
# 11. FINAL SUMMARY
# ============================================
print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)
print(f"Best model: {best_name} (Test R² = {best_r2:.4f})")
print(f"All files saved in: {current_dir}")
print("\nYou can now run the Streamlit app:")
print("streamlit run car_price_app.py")
