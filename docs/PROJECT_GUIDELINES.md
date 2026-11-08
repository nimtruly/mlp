# MLP Project Guidelines (2026 T3) — summary

Source: https://bsc-iitm.github.io/mlp-project/guidelines.html (also the *Essentials*,
*Milestones*, *Resources* and *FAQ* pages). Read on 2026-11-08. The website is the
authority; this is a condensed copy for quick reference.

## Registration (deadline Nov 1, 2026; −1 mark per week late)

1. Create a Kaggle account and join the competition (link in the registration form:
   https://forms.gle/feAcuNCyyLHWzo6u5).
2. In the competition's **Code** tab create a new notebook named
   **`<RollNo>-notebook-2026t3`** (e.g. `21f1001234-notebook-2026t3`).
3. Share it with **`iitmbscs2008p`** — *can view*, *Allow comments* on, notebook **Private**.
   Keep this access until the end of the term.
4. Build a first model (even a dummy) and make a submission **from that notebook**.
5. Submit the notebook link in the registration form and check the status link
   (refreshed every 48 h).

## Key dates

| Event | Date |
|---|---|
| Problem statement released | Sep 30, 2026 |
| Notebook-link / registration deadline | Nov 1, 2026 |
| Milestone 1 / 2 / 3 / 4 / 5 | Oct 11 / Oct 18 / Oct 25 / Nov 1 / Nov 8, 2026 (Sundays) |
| Cross cut-off (>0.70) — Jan 2026 & earlier, May 2026 theory | Nov 15, 2026 |
| Cross cut-off — Sep 2026 theory | Nov 22, 2026 |
| Both vivas done — Jan/May 2026 theory | Dec 6, 2026 |
| Both vivas done — Sep 2026 theory | Dec 20, 2026 |

The competition runs at most 6 weeks (45 days).

## Passing criteria (all four)

1. Registration completed by Nov 1, 2026.
2. Registered notebook's submission crosses **> 0.70** on the leaderboard before the deadline.
3. **Level 1 viva** cleared (one reattempt if you took it in Batch 1).
4. **Level 2 viva** ≥ 25/50 (no reattempt).

Plus: at least **2 of 5 milestones** submitted (a milestone counts only when you get the
response-copy email).

## Marks

**Final = NB + M + L + S + V** (out of 100; 105 with bonus for May/Sep 2026 theory students)

| Part | Marks | How |
|---|---|---|
| NB | 10 | Notebook quality, graded by viva examiner |
| M | 5 | All milestones completed |
| L | 30 | `30 × (your best − 0.70) / (top score − 0.70)` |
| S | 5 | 5 if public score in top 20 %; 0 if < 10 submissions; else `5 × (N − 10)/(N_max − 10)` |
| V | 50 | Viva |
| Bonus | 5 | May/Sep 2026 theory students in Level-2 viva Batch 1 (Batch 1 also allows a reattempt) |

So: **make at least 10 submissions** over the competition (max 5/day), and every extra
leaderboard point above 0.70 is worth marks.

## Competition rules

- Individual project. No external data. **CPU only** — no Kaggle GPU/TPU.
- Max **5 submissions/day**. Only submissions made from the registered **FinalNotebook**
  count; uploading `.csv`/`.zip` manually is not counted (and is listed as invalidating).
- Notebook must stay **private** and shared with the admin.
- All end-to-end work must be visible and executed **in the Kaggle notebook** — local runs
  and pretrained models are not evaluated.

### Allowed libraries

NumPy, Pandas, Matplotlib, Seaborn, Plotly, Scikit-learn, XGBoost, LightGBM, CatBoost,
Imblearn, SciPy, statsmodels, regex, Wordcloud, and Python built-ins.

**Not allowed:** TensorFlow, PyTorch, NLTK, **pickle**, AutoML frameworks, spaCy, word2vec,
textblob, etc.

### Instant fail / invalidation

Different notebook link than the one used for submissions; notebook not shared or made
public; plagiarism; unauthorised libraries; external data; manual file submissions; fewer
than 2 milestones; missing the cut-off or viva deadlines.

## Level 1 viva checklist (the notebook must cover all of it)

- [ ] Problem statement understood; dataset loaded correctly.
- [ ] End-to-end reasoning: EDA → preprocessing → feature engineering → modelling →
      error/performance analysis, explaining how each step shapes the next.
- [ ] **EDA:** key statistics, univariate + bivariate analysis, clear relevant plots.
- [ ] **Preprocessing:** missing-value detection/handling, scaling of numericals, encoding of
      categoricals — implemented with **reusable pipelines / functions applied identically to
      train and test** (no leakage).
- [ ] **Feature engineering:** new / transformed features and an analysis of their importance.
- [ ] **Models:** several model families + hyperparameter tuning.
- [ ] **Comparison:** at least **three** models compared on train/validation with proper
      metrics; pros and cons of each.
- [ ] Clean, structured, commented notebook with outputs.
- [ ] The notebook contains the **best-performing model**.

## Level 2 viva — what to prepare

- Explain every line of your code and why you made each choice.
- Questions can come from any diploma course (MLF, MLT, statistics …).
- For every model used, know the 5 components: **data, model, loss, optimisation, evaluation**.
- Possible live coding: write or modify code during the viva.
