# House price predictor

A full-stack machine learning project: a scikit-learn model, a Flask JSON API, and a plain HTML/CSS/JS frontend.

```
house-price-predictor/
├── backend/
│   ├── train.py          # builds data, trains the model, saves model/ (model.joblib + meta.json)
│   ├── app.py            # Flask API + serves the frontend
│   ├── tests/test_api.py # pytest checks for the API
│   └── requirements.txt
└── frontend/
    ├── index.html
    ├── style.css
    └── app.js            # reads the form, calls the API, shows results
```

## Run it

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python train.py                  # creates backend/model/
python app.py                    # http://127.0.0.1:5000
```

Open http://127.0.0.1:5000. The Flask app serves both the API and the frontend, so no extra setup or CORS config is needed.

Run the tests with `pytest` (inside `backend/`, after training).

## API

| Method | Path           | Purpose                                                                 |
|--------|----------------|-------------------------------------------------------------------------|
| GET    | `/api/meta`    | Localities, input limits, baseline home, model metrics, scatter points |
| POST   | `/api/predict` | Price, 80% range, price per sq ft, factor effects, locality comparison |

Example request:

```bash
curl -X POST http://127.0.0.1:5000/api/predict -H "Content-Type: application/json" \
  -d '{"locality":"Salt Lake","area_sqft":1100,"bedrooms":2,"bathrooms":2,"age_years":8,"furnishing":1,"parking":1}'
```

Invalid input returns HTTP 400 with `{"error": "..."}`.

## Use your own data

```bash
python train.py --csv my_houses.csv
```

Required columns: `locality, area_sqft, bedrooms, bathrooms, age_years, furnishing (0/1/2), parking (0/1), price (rupees)`.
Update the `LIMITS` in `train.py` if your data covers a wider range of areas or ages.

## Production

```bash
gunicorn -w 2 -b 0.0.0.0:8000 app:app
```

## How it works

- **Model:** gradient boosting regressor on the log of the price, with one-hot encoded locality. 80% of rows train, 20% test; the 10th and 90th percentile of test errors define the likely price range.
- **Explanations:** "What moves the price" resets one detail at a time to the average home and measures the change in predicted price.
- **Data:** the default dataset is synthetic, so predictions illustrate the workflow and are not real valuations.
