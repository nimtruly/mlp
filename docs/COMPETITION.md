# Competition: Workstation Rental Price Tier Prediction

Link: https://www.kaggle.com/competitions/workstation-rental-price-tier-prediction

> The competition page is only visible to logged-in participants, so the overview, data
> dictionary and evaluation metric could not be read automatically. Everything below marked
> **confirmed** comes from files we have; items marked **TODO** must be filled in from the
> Kaggle *Overview* and *Data* tabs or from `train.csv` once it is downloaded.

## Task

| Item | Value | Status |
|---|---|---|
| Problem type | Multi-class classification of `price_tier` | confirmed (integer target in sample submission) |
| Target column | `price_tier` (integer) | confirmed |
| ID column | `ID` | confirmed |
| Number of test rows | 114,538 (IDs 0 – 114537, sorted) | confirmed |
| Possible tier values | TODO — `train['price_tier'].value_counts()` | TODO |
| Are tiers ordered (low → high price)? | Very likely, by name; check against features | TODO |
| Evaluation metric | TODO — read the *Evaluation* section (accuracy / macro-F1 / QWK …) | TODO |
| Pass cut-off | Leaderboard score **> 0.70** | confirmed (guidelines) |
| Competition deadline | Nov 15, 2026 (Nov 22 for Sep-2026 theory students) | confirmed (guidelines) |

## Files

| File | Size | In repo? | Notes |
|---|---|---|---|
| `train.csv` | > 30 MB | no — download | features + `price_tier` |
| `test.csv` | > 30 MB | no — download | features + `ID`, no target |
| `sample_submission.csv` | 0.9 MB | `data/sample_submission.csv` | every row predicts tier `2` (a constant baseline) |

## Submission format

```csv
ID,price_tier
0,2
1,2
...
114537,2
```

Exactly 114,538 data rows, header `ID,price_tier`, integer tiers. The file must be written
by the registered Kaggle notebook (e.g. `submission.to_csv('submission.csv', index=False)`)
and submitted from the notebook's output — not uploaded by hand.

## First things to learn from the data (once downloaded)

Run these first; record answers in this file.

```python
import pandas as pd
train = pd.read_csv('data/train.csv')
test  = pd.read_csv('data/test.csv')

train.shape, test.shape
train.dtypes.value_counts()            # numeric vs categorical
train['price_tier'].value_counts(normalize=True)   # class balance → metric & CV choice
train.isna().mean().sort_values(ascending=False).head(20)   # missing values
train.nunique().sort_values()          # constant / ID-like / high-cardinality columns
set(train.columns) - set(test.columns) # should be just the target
```

Things to decide from those answers:

- **Class balance** → use `StratifiedKFold`; consider `class_weight` if skewed.
- **Ordinal target?** → if tiers are ordered, regression-then-round or ordinal-aware
  metrics may help (especially if the metric is QWK).
- **Categoricals** → one-hot for linear/KNN/SVM; native categorical handling in
  LightGBM/CatBoost; ordinal encoding is fine for tree models.
- **Size (~hundreds of thousands of rows)** → SVM/KNN on the full set will be slow on CPU;
  subsample for those, and favour linear models + gradient boosting for the full data.

## Data dictionary

TODO — copy from the Kaggle *Data* tab.

| Column | Type | Description |
|---|---|---|
| `ID` | int | row identifier |
| … | | |
| `price_tier` | int | target (train only) |
