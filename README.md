# Fashion Store Review Demo

A Flask coursework project that combines category search with a text-classification
demo for clothing-review recommendations.

## What it demonstrates

- category search using stemming, fuzzy matching, and an inverted index;
- a responsive product and review interface;
- inference with a stored text vectorizer and classifier; and
- a small JSON API used by the front end.

The repository is a demonstration application, not a production service. The
original training and evaluation notebook was not included, so this repository
does not make a verified accuracy claim for the stored model.

## Run locally

Use Python 3.10 or newer in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000` in a browser. Set `FLASK_DEBUG=1` only when
debugging locally.

## Data and model artefacts

- `assignment3_II.csv` is the coursework dataset used by the application.
- `model.pkl` and `vectorizer.pkl` are the stored classifier and text vectorizer.
  Their metadata identifies scikit-learn 1.5.1, which is pinned in
  `requirements.txt` to avoid cross-version loading problems.

The original download source, licence, package versions, training split, and
evaluation results were not recorded in this repository. They should be
established before the dataset or model is redistributed or used beyond this
coursework demonstration. Python pickle files can execute code when loaded; use
these artefacts only from a source you trust.

## Limitations

- reviews added through the interface are held in memory and are lost on restart;
- the application is intended for local demonstration rather than deployment;
- retraining and reproducible evaluation are not yet part of the repository; and
- no performance, fairness, or generalisation claim is made.
