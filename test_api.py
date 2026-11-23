import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import app  # noqa: E402

GOOD = dict(locality="Salt Lake", area_sqft=1100, bedrooms=2, bathrooms=2,
            age_years=8, furnishing=1, parking=1)


def test_predict_ok():
    r = app.test_client().post("/api/predict", json=GOOD)
    assert r.status_code == 200
    d = r.get_json()
    assert d["low"] < d["price"] < d["high"]
    assert len(d["comparison"]) == 8


def test_bigger_house_costs_more():
    c = app.test_client()
    a = c.post("/api/predict", json=GOOD).get_json()["price"]
    b = c.post("/api/predict", json={**GOOD, "area_sqft": 2000}).get_json()["price"]
    assert b > a


def test_validation():
    r = app.test_client().post("/api/predict", json={**GOOD, "locality": "Mars"})
    assert r.status_code == 400
