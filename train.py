"""Train the house price model and save it to backend/model/.

Usage:
    python train.py                # trains on a synthetic Kolkata-area dataset
    python train.py --csv my.csv   # trains on your own data

CSV columns required: locality, area_sqft, bedrooms, bathrooms, age_years,
furnishing (0=unfurnished, 1=semi, 2=full), parking (0/1), price (in rupees).
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

OUT = Path(__file__).parent / "model"
NUMERIC = ["area_sqft", "bedrooms", "bathrooms", "age_years", "furnishing", "parking"]
FEATURES = ["locality"] + NUMERIC

# Input limits enforced by the API. Widen these if your data covers more.
LIMITS = {"area_sqft": [350, 3500], "age_years": [0, 40],
          "bedrooms": [1, 5], "bathrooms": [1, 5]}

# Average rupees per sq ft by locality (used only to generate sample data)
LOCALITIES = {
    "Alipore": 18000, "Ballygunge": 14000, "Salt Lake": 8500, "New Town": 7000,
    "Dum Dum": 5500, "Behala": 5000, "Garia": 4800, "Howrah": 4200,
}


def synthetic(n=3000, seed=42):
    rng = np.random.default_rng(seed)
    loc = rng.choice(list(LOCALITIES), n)
    bed = rng.integers(1, 6, n)
    area = np.clip(np.round((380 + bed * 340 + rng.normal(0, 160, n)) / 25) * 25, 350, 3500)
    bath = np.clip(np.round(bed * 0.7 + 0.5 + rng.normal(0, 0.6, n)), 1, 5).astype(int)
    age = rng.integers(0, 41, n)
    furn = rng.integers(0, 3, n)
    park = rng.integers(0, 2, n)
    ppsf = np.array([LOCALITIES[l] for l in loc])
    price = (area * ppsf * (1 + 0.04 * (bath - 2)) * (1 - 0.009 * age)
             * np.where(park == 1, 1.06, 1.0) * (1 + 0.035 * furn)
             * np.exp(rng.normal(0, 0.07, n)))
    return pd.DataFrame(dict(locality=loc, area_sqft=area, bedrooms=bed, bathrooms=bath,
                             age_years=age, furnishing=furn, parking=park, price=price.round()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="optional CSV with real data")
    args = ap.parse_args()

    df = pd.read_csv(args.csv) if args.csv else synthetic()
    X, y = df[FEATURES], df["price"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)

    pre = ColumnTransformer(
        [("loc", OneHotEncoder(handle_unknown="ignore"), ["locality"])],
        remainder="passthrough",
    )
    model = TransformedTargetRegressor(
        regressor=Pipeline([
            ("pre", pre),
            ("gbr", GradientBoostingRegressor(n_estimators=400, max_depth=3,
                                              learning_rate=0.05, random_state=0)),
        ]),
        func=np.log, inverse_func=np.exp,
    )
    model.fit(X_tr, y_tr)

    pred = model.predict(X_te)
    log_err = np.log(y_te.values) - np.log(pred)
    lo_q, hi_q = np.quantile(log_err, [0.10, 0.90])  # 80% interval in log space

    # Baseline "average" house used to explain predictions
    by_loc = df.assign(p=df.price / df.area_sqft).groupby("locality").p.mean().sort_values()
    baseline = {c: int(round(df[c].median())) for c in NUMERIC}
    baseline["locality"] = by_loc.index[len(by_loc) // 2]

    meta = {
        "localities": list(by_loc.index[::-1]),  # most expensive first
        "baseline": baseline,
        "interval": {"lo": float(lo_q), "hi": float(hi_q)},
        "limits": LIMITS,
        "metrics": {
            "r2": float(r2_score(y_te, pred)),
            "mape": float(mean_absolute_percentage_error(y_te, pred)),
            "train_rows": int(len(X_tr)), "test_rows": int(len(X_te)),
        },
        "scatter": [{"actual": float(a), "predicted": float(p)}
                    for a, p in list(zip(y_te, pred))[:250]],
    }
    OUT.mkdir(exist_ok=True)
    joblib.dump(model, OUT / "model.joblib")
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved model. R2={meta['metrics']['r2']:.3f}  MAPE={meta['metrics']['mape']:.3%}")


if __name__ == "__main__":
    main()
