# Milestone 4: Answers (source for the filled .docx)

Generated from the notebook `notebooks/workstation_price_tier.ipynb`. All scores are **macro-F1**
on the same stratified 20% hold-out (55,950 rows) unless marked CV. Build the Word file with
`python docs/milestones/fill_m4_docx.py --roll <your-roll-number>`, then export it to PDF as
`<roll>_26t3_M4.pdf`.

## Q1. Model Building Plan
### Answer
TODO
### Code
```python
X_full, X_test = build_features(train_raw), build_features(test_raw)   # stateless feature engineering
X_tr, X_val, y_tr, y_val = train_test_split(X_full, y_full, test_size=0.2, stratify=y_full, random_state=42)

baseline_models = {
    'Logistic Regression': make_pipeline(make_linear_preprocessor(), LogisticRegression(max_iter=1000)),
    'KNN (50k sample)':    make_pipeline(make_linear_preprocessor(), KNeighborsClassifier()),
    'Decision Tree':       make_pipeline(make_tree_preprocessor(), DecisionTreeClassifier(random_state=42)),
    'Random Forest':       make_pipeline(make_tree_preprocessor(), RandomForestClassifier(n_estimators=100)),
    'LightGBM':            make_pipeline(make_tree_preprocessor(impute=False), lgb.LGBMClassifier()),
    'XGBoost':             make_pipeline(make_tree_preprocessor(impute=False),
                                         EncodeLabels(xgb.XGBClassifier(tree_method='hist'))),
}
for name, model in baseline_models.items():
    X_fit, y_fit = (X_tr.loc[knn_idx], y_tr.loc[knn_idx]) if name.startswith('KNN') else (X_tr, y_tr)
    model.fit(X_fit, y_fit)
    print(name, f1_score(y_val, model.predict(X_val), average='macro'))
```
### Figures
- Q1.1 | Baseline models: hold-out macro-F1 | notebooks/figures/Q1.1_baseline_model_comparison.png

## Q2. Linear Models
### Answer
TODO
### Code
```python
def make_linear_preprocessor():
    log_scale = Pipeline([('impute', SimpleImputer(strategy='median')),
                          ('log', FunctionTransformer(np.log1p)), ('scale', StandardScaler())])
    plain = Pipeline([('impute', SimpleImputer(strategy='median', add_indicator=True)), ('scale', StandardScaler())])
    return ColumnTransformer([
        ('target_enc', make_pipeline(TargetEncoder(random_state=42), StandardScaler()), HIGH_CARD),
        ('freq_enc', make_pipeline(FrequencyEncoder(), FunctionTransformer(np.log1p), StandardScaler()), FREQ_COLS),
        ('onehot', OneHotEncoder(handle_unknown='ignore'), LOW_CARD),
        ('log_num', log_scale, SKEWED), ('num', plain, PLAIN)])

lr_pipe = make_pipeline(make_linear_preprocessor(), LogisticRegression(max_iter=2000))
lr_grid = {'logisticregression__C': [0.001, 0.01, 0.1, 1, 10],
           'logisticregression__class_weight': [None, 'balanced']}
lr_search = GridSearchCV(lr_pipe, lr_grid, scoring='f1_macro', cv=StratifiedKFold(3, shuffle=True, random_state=42))
lr_search.fit(X_tune, y_tune)
# preprocessing ablation: no scaling / scaling without log / log + scaling
for label, pre in [('no scaling', no_scale), ('scaling, no log', no_log), ('log + scaling', make_linear_preprocessor())]:
    print(label, cross_val_score(lr_variant(pre), X_tune, y_tune, scoring='f1_macro', cv=CV3).mean())
best_lr = clone(lr_pipe).set_params(**lr_search.best_params_).fit(X_tr, y_tr)
```
### Figures
- Q2.1 | Logistic regression: CV macro-F1 vs C | notebooks/figures/Q2.1_logreg_C_validation_curve.png

## Q3. Other Non-Tree Models
### Answer
TODO
### Code
```python
knn_pipe = make_pipeline(make_linear_preprocessor(), KNeighborsClassifier(algorithm='brute'))
knn_grid = {'kneighborsclassifier__n_neighbors': [5, 11, 21, 41, 81],
            'kneighborsclassifier__weights': ['uniform', 'distance']}
knn_search = GridSearchCV(knn_pipe, knn_grid, scoring='f1_macro', cv=CV3).fit(X_tune.iloc[:30000], y_tune.iloc[:30000])

for n in [10000, 25000, 50000, 100000]:            # effect of training-set size
    idx = X_tr.sample(n, random_state=42).index
    m = clone(knn_pipe).set_params(**knn_search.best_params_).fit(X_tr.loc[idx], y_tr.loc[idx])
    print(n, f1_score(y_val, m.predict(X_val), average='macro'))
```
### Figures
- Q3.1 | KNN: CV macro-F1 vs k and weighting | notebooks/figures/Q3.1_knn_k_validation_curve.png

## Q4. Decision Trees
### Answer
TODO
### Code
```python
def make_tree_preprocessor(impute=True):
    return ColumnTransformer([
        ('target_enc', TargetEncoder(random_state=42), HIGH_CARD),
        ('freq_enc', FrequencyEncoder(), FREQ_COLS),
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), LOW_CARD),
        ('num', SimpleImputer(strategy='median') if impute else 'passthrough', NUM_COLS)])

dt_pipe = make_pipeline(make_tree_preprocessor(), DecisionTreeClassifier(random_state=42))
dt_grid = {'decisiontreeclassifier__max_depth': [6, 10, 14, 18, 24, None],
           'decisiontreeclassifier__min_samples_leaf': [1, 10, 50, 200],
           'decisiontreeclassifier__criterion': ['gini', 'entropy']}
dt_search = GridSearchCV(dt_pipe, dt_grid, scoring='f1_macro', cv=CV3, return_train_score=True).fit(X_tune, y_tune)
best_dt = clone(dt_pipe).set_params(**dt_search.best_params_).fit(X_tr, y_tr)
plot_tree(best_dt[-1], max_depth=2, feature_names=list(best_dt[0].get_feature_names_out()), filled=True)
```
### Figures
- Q4.1 | Decision tree: validation and training macro-F1 vs depth and leaf size | notebooks/figures/Q4.1_decision_tree_depth_leaf.png
- Q4.2 | Top three levels of the tuned decision tree | notebooks/figures/Q4.2_decision_tree_top_levels.png

## Q5. Bagging Ensembles
### Answer
TODO
### Code
```python
rf_pipe = make_pipeline(make_tree_preprocessor(), RandomForestClassifier(random_state=42))
rf_space = {'randomforestclassifier__n_estimators': [150, 300],
            'randomforestclassifier__max_features': ['sqrt', 0.2, 0.35],
            'randomforestclassifier__max_depth': [None, 20, 30],
            'randomforestclassifier__min_samples_leaf': [1, 2, 5],
            'randomforestclassifier__class_weight': [None, 'balanced_subsample']}
rf_search = RandomizedSearchCV(rf_pipe, rf_space, n_iter=12, scoring='f1_macro', cv=CV3, random_state=42)
rf_search.fit(X_tune, y_tune)

for n in [10, 25, 50, 100, 200, 300]:              # out-of-bag score vs number of trees
    rf_ = RandomForestClassifier(**dict(best_rf_params, n_estimators=n), oob_score=True).fit(Xt, y_tune)
    oob_pred = rf_.classes_[rf_.oob_decision_function_.argmax(1)]
    print(n, f1_score(y_tune, oob_pred, average='macro'))
```
### Figures
- Q5.1 | Random forest: out-of-bag macro-F1 vs number of trees | notebooks/figures/Q5.1_random_forest_n_estimators_oob.png
- Q5.2 | Random forest: top-20 impurity-based feature importances | notebooks/figures/Q5.2_random_forest_feature_importance.png

## Q6. Boosting Ensembles
### Answer
TODO
### Code
```python
lgb_pipe = make_pipeline(make_tree_preprocessor(impute=False), lgb.LGBMClassifier(subsample_freq=1, verbose=-1))
lgb_space = {'lgbmclassifier__n_estimators': [200, 400, 700], 'lgbmclassifier__learning_rate': [0.03, 0.06, 0.1],
             'lgbmclassifier__num_leaves': [31, 63, 127, 255], 'lgbmclassifier__min_child_samples': [10, 30, 80],
             'lgbmclassifier__subsample': [0.7, 0.85, 1.0], 'lgbmclassifier__colsample_bytree': [0.3, 0.5, 0.8],
             'lgbmclassifier__reg_lambda': [0, 1, 5]}
lgb_search = RandomizedSearchCV(lgb_pipe, lgb_space, n_iter=16, scoring='f1_macro', cv=CV3, random_state=42)
lgb_search.fit(X_tune, y_tune)

# early stopping on a 10% split of the TRAINING part chooses the number of trees per learning rate
X_in, X_es, y_in, y_es = train_test_split(X_tr, y_tr, test_size=0.1, stratify=y_tr, random_state=42)
for lr in [0.1, 0.05]:
    m = lgb.LGBMClassifier(**dict(best_lgb_params, learning_rate=lr, n_estimators=3000))
    m.fit(Xin_t, y_in, eval_set=[(Xes_t, y_es)], callbacks=[lgb.early_stopping(100)])
```
### Figures
- Q6.1 | LightGBM: early-stopping log-loss vs number of trees for two learning rates | notebooks/figures/Q6.1_lightgbm_learning_curve.png

## Q7. Voting and Stacking
### Answer
TODO
### Code
```python
vote = VotingClassifier([('lr', best_lr), ('rf', best_rf), ('lgb', lgb_es), ('xgb', best_xgb)], voting='soft')
vote.fit(X_tr, y_tr)
for params in [dict(voting='soft', weights=None), dict(voting='soft', weights=[1, 1, 2, 2]),
               dict(voting='soft', weights=[0, 1, 1, 1]), dict(voting='hard', weights=None)]:
    vote.set_params(**params)                      # used only at predict time: no refit needed
    print(params, f1_score(y_val, vote.predict(X_val), average='macro'))

stack = StackingClassifier(estimators=[('lr', best_lr), ('rf', best_rf), ('lgb', lgb_es)],
                           final_estimator=LogisticRegression(max_iter=2000),
                           cv=StratifiedKFold(3, shuffle=True, random_state=42), stack_method='predict_proba')
stack.fit(X_tr, y_tr)
print('stacking', f1_score(y_val, stack.predict(X_val), average='macro'))
```
### Figures
- Q7.1 | Base models vs voting and stacking (hold-out macro-F1) | notebooks/figures/Q7.1_voting_stacking_comparison.png

## Q8. Hyperparameter Tuning(HPT) Strategy
### Answer
TODO
### Code
```python
for label, cv in [('KFold k=5', KFold(5, shuffle=True, random_state=42)),
                  ('StratifiedKFold k=3', StratifiedKFold(3, shuffle=True, random_state=42)),
                  ('StratifiedKFold k=5', StratifiedKFold(5, shuffle=True, random_state=42)),
                  ('StratifiedKFold k=10', StratifiedKFold(10, shuffle=True, random_state=42))]:
    s = cross_val_score(best_lr, X_tune, y_tune, scoring='f1_macro', cv=cv)
    print(label, s.mean(), s.std())
# plus 5 repeated 80/20 hold-out splits for comparison

grid = GridSearchCV(dt_pipe, dt_grid, scoring='f1_macro', cv=CV3).fit(X_tune, y_tune)               # 48 configs
rand = RandomizedSearchCV(dt_pipe, dt_grid, n_iter=12, scoring='f1_macro', cv=CV3).fit(X_tune, y_tune)
halv = HalvingRandomSearchCV(dt_pipe, dt_grid, n_candidates=48, factor=3, min_resources=2000,
                             scoring='f1_macro', cv=CV3).fit(X_tune, y_tune)
```
### Figures
- Q8.1 | Cross-validation strategies (mean ± std) and search-strategy cost | notebooks/figures/Q8.1_cv_and_search_strategies.png

## Q9. Hyperparameter tuning Effects
### Answer
TODO
### Code
```python
X_tune, _, y_tune, _ = train_test_split(X_tr, y_tr, train_size=60000, stratify=y_tr, random_state=42)
CV3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
searches = {'Logistic Regression': lr_search,   # GridSearchCV, 10 configs
            'KNN': knn_search,                  # GridSearchCV, 10 configs (30k rows)
            'Decision Tree': dt_search,         # GridSearchCV, 48 configs
            'Random Forest': rf_search,         # RandomizedSearchCV, 12 of 162 configs
            'LightGBM': lgb_search,             # RandomizedSearchCV, 16 of 8,748 configs + early stopping
            'XGBoost': xgb_search}              # RandomizedSearchCV, 8 of 96 configs
for name, s in searches.items():
    print(name, s.best_params_, round(s.best_score_, 4))
```
### Figures
- Q9.1 | LightGBM random search: CV macro-F1 per sampled configuration | notebooks/figures/Q9.1_lightgbm_random_search.png

## Q10. Model Score Analysis
### Answer
TODO
### Code
```python
res = pd.DataFrame(RESULTS)                         # every evaluate() call logs model, stage, hold-out macro-F1
ba = res[res.stage.isin(['baseline', 'tuned'])]
before_after = ba.pivot_table(index='model', columns='stage', values='holdout_macro_f1', aggfunc='max')
before_after['gain'] = before_after['tuned'] - before_after['baseline']
print(before_after.sort_values('tuned'))
print(classification_report(y_val, best_model.predict(X_val), digits=4))
```
### Figures
- Q10.1 | Hold-out macro-F1 before vs after tuning | notebooks/figures/Q10.1_before_after_tuning.png
- Q10.2 | Confusion matrix of the best model on the hold-out | notebooks/figures/error_confusion_matrix.png
