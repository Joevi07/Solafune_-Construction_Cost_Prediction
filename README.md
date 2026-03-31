# Solafune_-Construction_Cost_Prediction
# Construction Cost Per m² Estimation
### Solafune Competition — Solution

![Score](https://img.shields.io/badge/Best%20RMSE-0.22129-brightgreen)
![Models](https://img.shields.io/badge/Ensemble-15%20Models-blue)
![Platform](https://img.shields.io/badge/Platform-Solafune-orange)

---

## Problem
Predict construction cost per square meter (USD) for locations across **Japan** and the **Philippines** using tabular economic/geographic features and satellite imagery. Evaluated by **RMSE on log1p-transformed targets**.

---

##  Dataset
- **Tabular**: GDP, distance to capital, infrastructure flags, climate zone, hazard indicators, year/quarter
- **Sentinel-2 L2A**: 12-band multispectral imagery (quarterly composites)
- **VIIRS**: Nighttime light radiance

- **evaluation_dataset.zip**
File type:zipFile count:2049Data description:Evaluation set of tabular data and composite data.
- **train_dataset.zip**
File type:zipFile count:2049Data description:Train set of tabular data and composite data.
- **sample_submission.csv**
File type:csvFile count:1Data description:Sample file for submit this is my dataset part actually what you have given is diff

---

##  Solution Overview

### 1. Image Features (Sentinel-2)
| Feature | Description |
|---------|-------------|
| `B8_mean` | Mean NIR reflectance — proxy for surface brightness |
| `NDVI` | (B8 − B4) / (B8 + B4) — proxy for urban density |

### 2. Tabular Feature Engineering
- Log-transformed GDP and distance to capital
- Interaction terms: `gdp × dist`, `developed × gdp`, `developed × dist`
- Bayesian target encoding for `geolocation_name` (α=25)
- Ordinal cyclone risk encoding, one-hot quarter dummies

### 3. Model
**CatBoostRegressor** on `log1p`-transformed target
- Depth: 7 | LR: 0.035 | Iterations: 1000 (early stopping: 100)

### 4. Ensemble
5-fold CV × 3 seeds (42, 123, 456) = **15 models**; averaged in log-space, clipped to training percentile range.

---

##  Results

| Metric | Value |
|--------|-------|
| Best Submission RMSE | **0.22129** |
| CV RMSE (mean) | ~0.221 |
| Ensemble Size | 15 models |
| Features Used | 13 (11 tabular + 2 image) |

---

##  Tech Stack
`Python` `CatBoost` `scikit-learn` `rasterio` `pandas` `numpy`

---

##  Usage

```bash
git clone https://github.com/your-repo
cd your-repo
pip install -r requirements.txt
python solution.py
```

Output saved to `outputs/submission.csv`.

---


