# Competition: Workstation Rental Price Tier Prediction

Link: https://www.kaggle.com/competitions/workstation-rental-price-tier-prediction

> The competition page is only visible to logged-in participants, so the overview text and
> the **evaluation metric** could not be read automatically. Everything else below comes from
> the data files themselves (downloaded 2026-11-08). Remaining **TODO**s need the Kaggle page.

## Task

| Item | Value |
|---|---|
| Problem type | **4-class classification** of `price_tier` (ordinal: 1 = cheapest … 4 = most expensive) |
| Target column | `price_tier` ∈ {1, 2, 3, 4} |
| ID column | `space_id` in the data files → written as **`ID`** in the submission |
| Train rows | 279,746 × 60 columns (59 features + target) |
| Test rows | 114,538 × 59 columns (`space_id` 0 – 114537) |
| Class balance | 1: 23.9 % · 2: 29.1 % · 3: 21.7 % · 4: 25.2 % — fairly balanced |
| Evaluation metric | **TODO** — read the *Evaluation* section (accuracy / macro-F1 / QWK …) |
| Pass cut-off | Leaderboard score **> 0.70** |
| Competition deadline | Nov 15, 2026 (Nov 22 for Sep-2026 theory students) |

The data reads like a short-term-rental listing dataset with renamed fields: "spaces"
(listings) run by "operators" (hosts), with location, capacity, booking terms, reviews and
facilities (amenities).

## Files

| File | Size | In git? | Notes |
|---|---|---|---|
| `data/train.csv` | 195 MB | no (git-ignored) | features + `price_tier` |
| `data/test.csv` | 80 MB | no (git-ignored) | features, no target |
| `data/sample_submission.csv` | 0.9 MB | yes | every row predicts tier `2` (constant baseline) |

## Submission format

```csv
ID,price_tier
0,2
1,2
...
114537,2
```

Exactly 114,538 data rows, header `ID,price_tier` (rename `space_id` → `ID`), integer tiers
1–4. Write it from the registered Kaggle notebook and submit from the notebook's output —
never upload it by hand.

## Data dictionary (inferred from the data)

Missing % is for train (test is within ~1 point).

| Group | Columns | Type | Notes |
|---|---|---|---|
| ID | `space_id` | int | row id — drop before modelling |
| Operator (host) | `operator_ref` | str, 41,903 values | high-cardinality host name; 0.2 % missing |
| | `operator_platform_age_years/_months`, `operator_active_years/_months` | float | tenure split into years + months → combine into one number |
| | `operator_class` | cat (3) | unverified / independent / managed operator |
| | `operator_certified`, `operator_has_avatar`, `operator_id_verified` | t/f | binary flags |
| | `operator_portfolio_size`, `operator_metro_units/_whole/_private/_shared` | num | how many spaces the operator runs (in total / in this metro by type) |
| Location | `metro` | cat (16) | MTR-01 … MTR-16 |
| | `zone` | cat (1,380) | high-cardinality → target / frequency encoding |
| | `borough` | cat (55) | **76.7 % missing** |
| | `geo_lat`, `geo_lon` | float | coordinates (shifted/obfuscated ranges) |
| Space | `space_class` | cat (131) | e.g. "Boutique Workroom" |
| | `access_type` | cat (4) | whole_unit 75 %, private_zone 23 %, shared_zone, managed_suite |
| | `max_seats` | int | capacity — strongest-looking price signal |
| | `workstations`, `zones`, `utility_rooms` | float | 6.7 % / 17.3 % / 10.7 % missing |
| | `facilities` | str (JSON list) | e.g. `["FCL-087", "FCL-007", …]`, 0–101 codes (median 30) → count + multi-hot of common codes |
| Booking terms | `min_term_days/_floor/_ceiling/_avg`, `max_term_days/_floor/_ceiling/_avg` | float | min/max rental term; max values have large outliers |
| Availability | `open_slots_a … _e` | int | free slots over increasing horizons (≈30/60/90/365 days …) |
| | `booked_load` | int | |
| | `bookable` | t/f | instant-bookable flag (99 % t) |
| Reviews | `review_count`, `review_count_12m`, `review_count_30d`, `review_count_prev_yr` | int | |
| | `reviews_per_month`, `first_review_date`, `last_review_date` | | 19.4 % missing = never reviewed (same rows) |
| | `score_overall, _accuracy, _condition, _onboarding, _support, _access, _value` | float | 0–10 ratings, missing when no reviews |
| Compliance | `compliance_status` | cat (4) | unlicensed / licensed / exempt / in_review |
| Snapshot | `snapshot_start_date`, `snapshot_end_date` | date | scrape window, 2024-08 to 2025-08 |
| **Target** | `price_tier` | int 1–4 | train only |

### First observations

- Median `max_seats` rises with tier: 2, 2, 4, 5; `workstations`: 1, 1, 2, 3. Capacity separates
  tiers 3–4 from 1–2; tiers 1 vs 2 will need location/facilities/reviews.
- Review-related missingness is structural (19.4 % never reviewed) → add a `has_reviews` flag
  rather than just imputing.
- Dates should become numbers: days since first/last review relative to `snapshot_end_date`.
- 1,116 training rows are exact duplicates apart from `space_id`.

## Quick profiling snippet

Code used for the facts above; rerun after any data change.

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
