"""Flask API + static file server for the house price predictor."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

from train import FEATURES, OUT as MODEL_DIR

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
app = Flask(__name__, static_folder=None)

model = joblib.load(MODEL_DIR / "model.joblib")
META = json.loads((MODEL_DIR / "meta.json").read_text())
LIM = META["limits"]
LABELS = {"locality": "Locality", "area_sqft": "Area", "bedrooms": "Bedrooms",
          "bathrooms": "Bathrooms", "age_years": "Building age",
          "furnishing": "Furnishing", "parking": "Parking"}


def validate(data):
    """Return (clean_dict, error_message)."""
    if not isinstance(data, dict):
        return None, "Send a JSON object."
    if data.get("locality") not in META["localities"]:
        return None, f"locality must be one of: {', '.join(META['localities'])}."
    out = {"locality": data["locality"]}
    ranges = {**LIM, "furnishing": [0, 2], "parking": [0, 1]}
    for key, (lo, hi) in ranges.items():
        try:
            v = float(data[key])
        except (KeyError, TypeError, ValueError):
            return None, f"{key} is required and must be a number."
        if not lo <= v <= hi:
            return None, f"{key} must be between {lo} and {hi}."
        out[key] = v
    return out, None


def predict_many(rows):
    return model.predict(pd.DataFrame(rows)[FEATURES])


@app.get("/api/meta")
def meta():
    return jsonify({k: META[k] for k in ("localities", "baseline", "limits", "metrics", "scatter")})


@app.post("/api/predict")
def predict():
    row, err = validate(request.get_json(silent=True))
    if err:
        return jsonify(error=err), 400

    base = META["baseline"]
    # One batch: the house itself, each detail reset to the average, each locality
    variants = [row]
    for k in FEATURES:
        variants.append({**row, k: base[k]})
    variants += [{**row, "locality": loc} for loc in META["localities"]]
    p = predict_many(variants)

    price = float(p[0])
    factors = sorted(
        ({"name": LABELS[k], "effect": float(price - p[i + 1])} for i, k in enumerate(FEATURES)),
        key=lambda f: -abs(f["effect"]),
    )
    comparison = [{"locality": loc, "price": float(v)}
                  for loc, v in zip(META["localities"], p[1 + len(FEATURES):])]
    iv = META["interval"]
    avg_ppsf = float(predict_many([base])[0]) / base["area_sqft"]
    return jsonify(
        price=price,
        low=float(price * np.exp(iv["lo"])),
        high=float(price * np.exp(iv["hi"])),
        per_sqft=price / row["area_sqft"],
        vs_average_pct=(price / row["area_sqft"] / avg_ppsf - 1) * 100,
        factors=factors,
        comparison=comparison,
    )


@app.get("/")
def index():
    return send_from_directory(FRONTEND, "index.html")


@app.get("/<path:name>")
def assets(name):
    return send_from_directory(FRONTEND, name)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
