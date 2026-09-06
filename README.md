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

## Data and model artefacts

- `assignment3_II.csv` is the coursework dataset used by the application.
- `model.pkl` is a stored scikit-learn logistic-regression classifier with
  classes `[0, 1]`.
- `vectorizer.pkl` is a stored scikit-learn `CountVectorizer`.
- `ARTIFACTS.sha256` records SHA-256 digests for the dataset, model, and
  vectorizer. Check the tested snapshot with:

  ```bash
  shasum -a 256 -c ARTIFACTS.sha256
  ```

The original download source, redistribution licence, training split, and
evaluation results were not retained. That historical gap fixes the scope of
this repository: it demonstrates the preserved local application and inference
path, but it is not evidence for model quality and is not a basis for further
redistribution of the dataset or model. No open-source licence is asserted for
those artefacts. Python pickle files can execute code when loaded; use these
files only from a source you trust.

## Limitations

- reviews added through the interface are held in memory and are lost on restart;
- the application is intended for local demonstration rather than deployment;
- the retained snapshot covers inference and application behaviour, not model
  training or quality evaluation; and
- no performance, fairness, or generalisation claim is made.
