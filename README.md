# MLP Project — Workstation Rental Price Tier Prediction

IIT Madras BS — Machine Learning Practice (MLP) project, term **2026 T3**.

- **Kaggle competition:** https://www.kaggle.com/competitions/workstation-rental-price-tier-prediction
- **Project guidelines:** https://bsc-iitm.github.io/mlp-project/guidelines.html
- **Milestone 4 template (Google Doc):** https://docs.google.com/document/d/1bHIC9h2SBsSaM69-D6cefPKN03IO7XNHHBQFNVTm6rU/edit?tab=t.0

## The problem in one paragraph

Predict the **`price_tier`** (an integer class label) of a workstation rental from its
features. It is a **multi-class classification** problem: each row of `test.csv` gets
one tier, and the submission is a two-column CSV `ID,price_tier`. The test set has
**114,538 rows** (IDs `0`–`114537`). A leaderboard score **above 0.70** is required to
be eligible for the viva. See [docs/COMPETITION.md](docs/COMPETITION.md) for what we know
about the data and what still has to be confirmed once `train.csv` is loaded.

## Repository layout

```
.
├── README.md                     ← you are here
├── requirements.txt              ← only libraries the course allows
├── data/
│   ├── README.md                 ← how to get train.csv / test.csv
│   ├── sample_submission.csv     ← from Kaggle (committed, ~0.9 MB)
│   ├── train.csv                 ← NOT committed (>30 MB) — download it
│   └── test.csv                  ← NOT committed (>30 MB) — download it
├── scripts/
│   └── download_data.py          ← pulls the competition files with the Kaggle API
├── notebooks/                    ← project notebooks go here
├── docs/
│   ├── COMPETITION.md            ← problem, data, metric, submission format
│   ├── PROJECT_GUIDELINES.md     ← rules, deadlines, marks, vivas (summarised)
│   └── milestones/
│       ├── MILESTONES.md         ← all five milestones + submission rules
│       ├── MILESTONE_4.md        ← M4 questions as a working checklist
│       ├── Milestone_4_….pdf     ← original template (PDF)
│       └── Milestone_4_….docx    ← original template (DOCX)
└── ka3_*.ipynb, submission.csv   ← older work from a different Kaggle assignment (see below)
```

## Getting the data

`train.csv` and `test.csv` are too large to upload through the chat / GitHub web UI, so
they are git-ignored and downloaded locally. Full steps in [data/README.md](data/README.md);
short version:

```bash
pip install -r requirements.txt kaggle
# put your API token in ~/.kaggle/kaggle.json (Kaggle → Settings → API → Create New Token)
python scripts/download_data.py
```

On Kaggle itself (where the final notebook must run) the files are already mounted under
`/kaggle/input/workstation-rental-price-tier-prediction/`.

## Important rules (short list — read docs/PROJECT_GUIDELINES.md)

- Individual work; **no external data**; **CPU only** (no Kaggle accelerators).
- Only these libraries: NumPy, Pandas, Matplotlib, Seaborn, Plotly, Scikit-learn,
  XGBoost, LightGBM, CatBoost, Imblearn, SciPy, statsmodels, regex, Wordcloud.
  **Not allowed:** TensorFlow, PyTorch, NLTK, pickle, AutoML, spaCy, etc.
- Submissions only count when made **from the registered Kaggle notebook**
  (`<rollno>-notebook-2026t3`, private, shared with `iitmbscs2008p`). Uploading a CSV by hand
  is not counted and can invalidate the project. Max 5 submissions / day.
- Cut-off: leaderboard score **> 0.70** by **Nov 15, 2026** (Nov 22 for Sep 2026 theory term).

## Older files in this repo

`ka3_1.ipynb`, `ka3_1_!.ipynb`, `ka3_2.ipynb`, `ka3_2_1.ipynb` and `submission.csv` belong to
an earlier Kaggle assignment (`may-2026kaggle-assignment-3`, target column `class`), not to
this competition. `ka3_2.ipynb` and `ka3_2_1.ipynb` are identical, and **`ka3_1.ipynb`
contains unresolved git merge-conflict markers** (`<<<<<<< HEAD`), so it will not open in
Jupyter until that is fixed. They are left untouched.
