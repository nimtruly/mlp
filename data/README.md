# Data

| File | Committed? | How to get it |
|---|---|---|
| `sample_submission.csv` | yes | already here |
| `train.csv` | no (git-ignored, 195 MB) | download — see below |
| `test.csv` | no (git-ignored, 80 MB) | download — see below |

## Option A — Kaggle API (recommended)

1. On Kaggle: **Settings → API → Create New Token**. This downloads `kaggle.json`.
2. Put it at `~/.kaggle/kaggle.json` and run `chmod 600 ~/.kaggle/kaggle.json`
   (or set `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables).
3. You must have **joined the competition** (accepted its rules) with the same account.
4. From the repo root:

   ```bash
   pip install kaggle
   python scripts/download_data.py
   ```

## Option B — team Google Drive copy

```bash
curl -L -o data/train.csv "https://drive.usercontent.google.com/download?id=1PqndfpIbaVWqLKrxGWBtrUP6MuoenkVy&export=download&confirm=t"
curl -L -o data/test.csv  "https://drive.usercontent.google.com/download?id=14GPVoRqN7JVzbffqPUH3FayfND_5BHXn&export=download&confirm=t"
```

Expected sizes: `train.csv` 194,962,516 bytes, `test.csv` 79,849,746 bytes.

## Option C — manual

Download the zip from the competition's **Data** tab
(https://www.kaggle.com/competitions/workstation-rental-price-tier-prediction/data),
unzip, and place `train.csv` and `test.csv` in this folder.

## On Kaggle

Inside the competition notebook the files are already available; read them with:

```python
import os
DATA_DIR = '/kaggle/input/workstation-rental-price-tier-prediction'
if not os.path.exists(DATA_DIR):      # running locally
    DATA_DIR = 'data'
train = pd.read_csv(f'{DATA_DIR}/train.csv')
test  = pd.read_csv(f'{DATA_DIR}/test.csv')
```

(Check the exact mount path in the notebook's *Input* panel; newer Kaggle images use
`/kaggle/input/competitions/<slug>/`.)

Remember: **no external data** may be used for training.
