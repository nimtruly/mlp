# Milestone 4 — Model Building and Hyperparameter Exploration

- Template: [PDF](Milestone_4_Model_Building_and_Hyperparameter_Tuning.pdf) ·
  [DOCX](Milestone_4_Model_Building_and_Hyperparameter_Tuning.docx) ·
  [Google Doc](https://docs.google.com/document/d/1bHIC9h2SBsSaM69-D6cefPKN03IO7XNHHBQFNVTm6rU/edit?tab=t.0)
- Deliverable: `<roll-number>_26t3_M4.pdf`
- Rules: see [MILESTONES.md](MILESTONES.md#submission-rules-all-milestones). Each answer
  ~100–400 words, code pasted as text, figures at the end as `Figure Q<n>.<k>`.
- Cover **at least 50 % of the feature columns** across the milestone's answers.

Status: **not started** — waiting for `train.csv`/`test.csv` and the evaluation metric
(see [../COMPETITION.md](../COMPETITION.md)).

---

## Q1. Model Building Plan
Identify suitable models and explain the planned approach.
- [ ] Name suitable **linear, non-tree, tree-based and ensemble** models.
- [ ] Justify each from the target (multi-class `price_tier`, possibly ordinal), dataset
      size/feature types, and model assumptions.

## Q2. Linear Models
Implement ≥ 1 linear model (e.g. multinomial `LogisticRegression`).
- [ ] Assumptions and how it maps features → target (linear decision boundaries, log-odds).
- [ ] Effect of preprocessing (scaling, one-hot encoding, imputation); strengths/limits here.
- [ ] Hyperparameters: `C`, `penalty`, `solver`, `class_weight`, `max_iter` — and their effect.

## Q3. Other Non-Tree (non-linear) Models
≥ 1 model, e.g. **KNN**, **SVM (RBF)**, or **Gaussian Naive Bayes**.
- [ ] Assumptions and mechanism (distance / kernel / conditional independence).
- [ ] Preprocessing impact (scaling is critical for KNN/SVM); strengths/limits (cost on large n).
- [ ] Hyperparameters: KNN `n_neighbors`, `weights`, `metric`; SVM `C`, `gamma`, `kernel`.

## Q4. Decision Trees
- [ ] How a tree splits (Gini/entropy, greedy recursive partitioning) and represents rules.
- [ ] Preprocessing impact (no scaling needed; encoding choice); overfitting risk.
- [ ] Hyperparameters: `max_depth`, `min_samples_split`, `min_samples_leaf`, `criterion`,
      `ccp_alpha`, `max_features`.

## Q5. Bagging Ensembles
e.g. **RandomForestClassifier**, ExtraTrees, `BaggingClassifier`.
- [ ] Bootstrap sampling + feature subsampling + majority/average vote → lower variance.
- [ ] Preprocessing impact; strengths/limits (memory/time, less interpretable).
- [ ] Hyperparameters: `n_estimators`, `max_depth`, `max_features`, `min_samples_leaf`,
      `bootstrap`, `class_weight`.

## Q6. Boosting Ensembles
e.g. **LightGBM**, XGBoost, CatBoost, `HistGradientBoostingClassifier`.
- [ ] Sequential fitting of trees to the gradient of the loss (multi-class log-loss).
- [ ] Preprocessing impact (native missing/categorical handling); strengths/limits.
- [ ] Hyperparameters: `learning_rate`, `n_estimators`, `num_leaves`/`max_depth`,
      `subsample`, `colsample_bytree`, `reg_lambda`/`reg_alpha`, early stopping.

## Q7. Voting and Stacking
- [ ] Hard vs soft voting; stacking with out-of-fold predictions + meta-learner.
- [ ] Preprocessing per base model (pipelines); when ensembling helps (diverse errors).
- [ ] Hyperparameters: `voting`, `weights`, `final_estimator`, `cv`, `passthrough`.

## Q8. Hyperparameter Tuning Strategy
- [ ] What to consider when choosing hyperparameters and ranges; complexity vs
      generalisation (bias–variance).
- [ ] CV strategies: hold-out, k-fold, **stratified k-fold** (use this for class labels),
      repeated CV; why and how.
- [ ] Search methods: Grid, Random, Halving (successive halving), Bayesian (not allowed via
      external libs like Optuna — stick to sklearn) — coverage vs cost.

## Q9. Hyperparameter Tuning Effects
Tune **≥ 3 models**.
- [ ] Important hyperparameters + search spaces, with reasoning.
- [ ] Search strategy per model, justified by space size and compute (CPU only).
- [ ] CV strategy per model.
- [ ] Best configuration + CV score per model; observations (e.g. validation curves).

## Q10. Model Score Analysis
- [ ] Table of scores **before vs after** tuning for each model (same metric, same CV).
- [ ] Which models improved, by how much, and why.

---

## Figures checklist
Figure placeholders exist in the template for every question (Q1.1 … Q10.1). Useful ones:
- Q2/Q3/Q4: validation curve of one key hyperparameter (e.g. `C`, `n_neighbors`, `max_depth`).
- Q5/Q6: score vs `n_estimators` / learning-rate; feature importances.
- Q7: base-model vs ensemble score bar chart.
- Q9: CV score across the search (heatmap or line plot).
- Q10: before/after tuning bar chart.

> **Library note:** only the course-approved libraries may be used (sklearn, XGBoost,
> LightGBM, CatBoost, imblearn, …). Optuna/Hyperopt are not on the list — use
> `GridSearchCV`, `RandomizedSearchCV` or `HalvingRandomSearchCV`.
