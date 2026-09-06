# Fashion Store Review Demo

A Flask coursework project that combines category search with a text-classification
demo for clothing-review recommendations.

## What it demonstrates

- category search using plural normalization, fuzzy matching, and an inverted
  index;
- a responsive product and review interface;
- inference with a stored text vectorizer and classifier; and
- a small JSON API used by the front end.

The repository is a demonstration application, not a production service. The
original training and evaluation notebook was not included, so this repository
does not make a verified accuracy claim for the stored model.

## Run locally

The retained environment was verified with Python 3.12.7. `requirements.txt`
lists the three direct dependencies; `requirements.lock` records the complete
tested environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python app.py
```

Open `http://127.0.0.1:5000` in a browser. Set `FLASK_DEBUG=1` only when
debugging locally.

Run the API and model smoke tests from the repository root:

```bash
python -m unittest discover -s tests -v
```

The tests load the retained artefacts and exercise the home page, item listing,
category search, item details, classification, input validation, and the
in-memory review round trip. GitHub Actions runs the same suite on Python 3.12.

## Repository guide

| File or folder | Purpose |
|---|---|
| `app.py` | Flask routes, category search, model inference, and in-memory reviews |
| `templates/index.html` | Browser interface and API requests |
| `tests/test_app.py` | Application and API smoke tests |
| `assignment3_II.csv` | Clothing and review records displayed by the app |
| `model.pkl` and `vectorizer.pkl` | Stored classifier and text transformation |
| `requirements.lock` | Exact dependency versions for the retained environment |
| `ARTIFACTS.sha256` | Checksums for the retained data and model files |

## JSON API

| Method and path | Purpose |
|---|---|
| `GET /api/items?search=dress` | Search clothing categories; return up to 50 items |
| `GET /api/item/<item_id>` | Show an item and its review summary |
| `POST /api/predict` | Predict a recommendation from review text |
| `POST /api/reviews` | Add a review to the running application's memory |
| `GET /api/review/<review_id>` | Retrieve an added review |

With the app running, try category search and prediction:

```bash
curl 'http://127.0.0.1:5000/api/items?search=dress'

curl -X POST http://127.0.0.1:5000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"review_title":"Comfortable","review_text":"A comfortable and useful item."}'
```

Prediction requires a nonempty `review_text` string. `review_title` is optional
and must be a string. The response includes a binary `recommendation` and the
model's probability for the recommended class; this score is not a verified
measure of predictive accuracy.

To add a review, send an object to `/api/reviews` containing an existing
`item_id`, a nonempty `description`, a `rating` from 1 to 5, and a binary
`recommendation` (0 or 1). An optional `title` must be a string. The response
includes a `review_url` for retrieving the added review. Use an item ID returned
by `/api/items`.

Both POST endpoints expect a JSON object. Arrays, strings, scalar values,
malformed JSON, and invalid fields return HTTP 400 with an `error` message.
Unknown item or review IDs return HTTP 404.

## Data and model artefacts

- `assignment3_II.csv` is the retained coursework dataset used by the
  application. Its first ten fields and all 19,662 review rows are an exact
  row-level subset of version 1 of Kaggle's
  [Women's E-Commerce Clothing Reviews](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews)
  dataset (23,486 rows). Kaggle's official metadata identifies that upstream
  dataset as `CC0: Public Domain`.
- `model.pkl` is a stored scikit-learn logistic-regression classifier with
  classes `[0, 1]`.
- `vectorizer.pkl` is a stored scikit-learn `CountVectorizer`.
- `ARTIFACTS.sha256` records SHA-256 digests for the dataset, model, and
  vectorizer. Check the tested snapshot on macOS with:

  ```bash
  shasum -a 256 -c ARTIFACTS.sha256
  ```

  On Linux, use `sha256sum --check ARTIFACTS.sha256`.

The comparison was repeated on 6 September 2026 against the upstream file
`Womens Clothing E-Commerce Reviews.csv` (SHA-256
`bd93cc515747ad1f87b8bc863c9e40da758509e6db6506a96d06976e61e36ee0`).
The coursework file also contains `Clothes Title` and `Clothes Description`
display fields that do not occur in the upstream dataset; their original
construction was not recorded. The model-training notebook, split, and
evaluation results were also not retained.

The CC0 statement therefore applies only to the matched upstream review fields.
No separate licence is asserted for the two added display fields or the stored
model artefacts. This repository demonstrates the preserved local application
and inference path; it is not evidence for model quality. Python pickle files
can execute code when loaded, so use them only from a source you trust.

## Limitations

- reviews added through the interface are held in memory and are lost on restart;
- the application is intended for local demonstration rather than deployment;
- the retained snapshot covers inference and application behaviour, not model
  training or quality evaluation; and
- no performance, fairness, or generalisation claim is made.
