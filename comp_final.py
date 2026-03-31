import pandas as pd
import numpy as np
import rasterio
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")
# PATHS
BASE = Path(__file__).resolve().parent

TRAIN_TAB = BASE / "train_dataset/train_tabular.csv"
EVAL_TAB  = BASE / "evaluation_dataset/evaluation_tabular_no_target.csv"

TRAIN_IMG = BASE / "train_dataset/train_composite"
EVAL_IMG  = BASE / "evaluation_dataset/evaluation_composite"

OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)
# LOAD DATA
train = pd.read_csv(TRAIN_TAB)
test  = pd.read_csv(EVAL_TAB)

train["is_train"] = 1
test["is_train"] = 0
test["construction_cost_per_m2_usd"] = np.nan

df = pd.concat([train, test], ignore_index=True)
# IMAGE FEATURES
def img_features(row):
    img_dir = TRAIN_IMG if row.is_train else EVAL_IMG
    s2 = img_dir / f"sentinel_2_{row.data_id}_{row.geolocation_name}_{row.year}-{row.quarter_label}.tif"

    out = {}
    try:
        with rasterio.open(s2) as src:
            b4 = src.read(4).astype("float32")
            b8 = src.read(8).astype("float32")

            mask = (b4 > 0) & (b8 > 0)
            b4, b8 = b4[mask], b8[mask]

            if len(b4) > 0:
                out["B8_mean"] = np.mean(b8)
                out["NDVI"] = np.mean((b8 - b4) / (b8 + b4 + 1e-6))
    except:
        out["B8_mean"] = np.nan
        out["NDVI"] = np.nan

    return out

print("Extracting image features...")

with ThreadPoolExecutor(max_workers=6) as ex:
    futures = {ex.submit(img_features, r): i for i, r in df.iterrows()}
    for f in as_completed(futures):
        i = futures[f]
        for k, v in f.result().items():
            df.loc[i, k] = v
# TABULAR FEATURES
df["log_gdp"] = np.log1p(df["deflated_gdp_usd"])
df["log_dist"] = np.log1p(df["straight_distance_to_capital_km"])
df["gdp_x_dist"] = df["log_gdp"] * df["log_dist"]
df["inv_dist"] = 1 / (df["straight_distance_to_capital_km"] + 1)

df["developed"] = (df["developed_country"] == "Yes").astype(int)
df["developed_x_gdp"] = df["developed"] * df["log_gdp"]
df["developed_x_dist"] = df["developed"] * df["log_dist"]

for q in ["Q1","Q2","Q3","Q4"]:
    df[f"is_{q}"] = (df["quarter_label"] == q).astype(int)

risk_map = {'Very Low': 1, 'Low': 2, 'Moderate': 3, 'High': 4, 'Very High': 5}
df["cyclone_risk"] = df["tropical_cyclone_wind_risk"].map(risk_map).fillna(2)
df["developed_x_risk"] = df["developed"] * df["cyclone_risk"]
# LOCATION ENCODING
tr = df[df.is_train == 1]
target = "construction_cost_per_m2_usd"
global_mean = tr[target].mean()

alpha = 25
stats = tr.groupby("geolocation_name")[target].agg(["mean","count"])
geo_enc = (stats["count"] * stats["mean"] + alpha * global_mean) / (stats["count"] + alpha)
df["geo_enc"] = df["geolocation_name"].map(geo_enc).fillna(global_mean)
# FINAL CLEAN
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in num_cols:
    if col not in ['is_train', 'construction_cost_per_m2_usd', 'data_id']:
        df[col].fillna(df[col].median(), inplace=True)

train_df = df[df.is_train == 1].copy()
test_df  = df[df.is_train == 0].copy()
y = np.log1p(train_df[target].values)
# PREPARE FEATURES
exclude_cols = [
    'construction_cost_per_m2_usd', 'is_train', 'data_id',
    'geolocation_name', 'quarter_label', 'sentinel2_tiff_file_name',
    'viirs_tiff_file_name', 'country', 'region_economic_classification',
    'koppen_climate_zone', 'seismic_hazard_zone', 'flood_risk_class',
    'tropical_cyclone_wind_risk', 'tornadoes_wind_risk',
    'access_to_airport', 'access_to_port', 'access_to_highway', 'access_to_railway',
    'developed_country', 'landlocked'
]

feature_cols = [col for col in train_df.columns if col not in exclude_cols]
X_train = train_df[feature_cols].copy().astype(float)
X_test  = test_df[feature_cols].copy().astype(float)

print(f"Training with {len(feature_cols)} features")
# ENSEMBLE WITH MULTIPLE SEEDS
n_folds = 5
seeds   = [42, 123, 456]

all_fold_predictions = []
all_cv_scores = []

print("Training ensemble with multiple seeds...")

for seed_idx, seed in enumerate(seeds):
    print(f"\nSeed {seed} ({seed_idx + 1}/{len(seeds)})")
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_predictions = []
    seed_cv_scores   = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = CatBoostRegressor(
            iterations=1000, depth=7, learning_rate=0.035,
            loss_function="RMSE", bagging_temperature=0.6,
            random_strength=1.0, early_stopping_rounds=100,
            random_seed=seed, verbose=False
        )
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)

        val_pred = model.predict(X_val)
        score = np.sqrt(mean_squared_error(y_val, val_pred))
        seed_cv_scores.append(score)
        fold_predictions.append(model.predict(X_test))

    all_fold_predictions.append(np.mean(fold_predictions, axis=0))
    all_cv_scores.append(np.mean(seed_cv_scores))
    print(f"  Mean CV: {np.mean(seed_cv_scores):.6f}")

# FINAL ENSEMBLE + PREDICTION

final_test_preds = np.mean(all_fold_predictions, axis=0)
print(f"\nEnsemble CV: {np.mean(all_cv_scores):.6f}")

pred = np.expm1(final_test_preds)

train_target = train_df[target].values
q01 = np.percentile(train_target, 1)
q99 = np.percentile(train_target, 99)
pred = np.clip(pred, q01 * 0.95, q99 * 1.05)

submission = pd.DataFrame({
    "data_id": test_df["data_id"].values,
    "construction_cost_per_m2_usd": pred
})

submission.to_csv(OUT / "submission.csv", index=False)
print(f"\nDONE — outputs/submission.csv")
print(f"Predictions: min={pred.min():.2f}, max={pred.max():.2f}, mean={pred.mean():.2f}")
