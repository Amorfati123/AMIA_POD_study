# Domain-Based Stacked Ensemble for Postoperative Delirium Prediction

A two-stage stacked ensemble model for predicting postoperative delirium (POD) using preoperative and intraoperative EHR data. The model organizes 87 clinical features into three clinician-informed domains - patient-related, surgery-related, and anesthetic-related - trains domain-specific LightGBM base learners, and combines their outputs through a logistic regression meta-learner.

Developed using a retrospective case-control cohort of 5,729 surgical encounters from the Indiana Network for Patient Care (2012–2022).

## Repository Structure

```
├── PODelirium_Baselines.ipynb            # Baseline models
├── PODelirium_Stacked_Ensemble.ipynb     # Domain-based stacked ensemble
├── LICENSE
└── README.md
```

## Notebooks

### `PODelirium_Baselines.ipynb`

End-to-end pipeline for six baseline classifiers:

- **Data preprocessing** - column cleanup, medication class binarization, ASA class engineering, Phik-based correlation screening, missing value imputation
- **Feature schema** - automated classification of features into continuous, categorical, and binary types with JSON export for reproducibility
- **Unfitted preprocessor** - median imputation + robust scaling (continuous), mode imputation + one-hot encoding (categorical), mode imputation (binary), all fitted within CV folds to prevent leakage
- **Baseline models** - L2 Logistic Regression, Elastic Net Logistic Regression, Random Forest, Histogram Gradient Boosting, XGBoost (light tuning), XGBoost (expanded tuning + isotonic calibration)
- **Evaluation** - StratifiedGroupKFold (grouped by patient ID) with out-of-fold AUROC, PR-AUC, and Brier score
- **Interpretability** - global feature importances and SHAP beeswarm/bar plots for tree-based models

### `PODelirium_Stacked_Ensemble.ipynb`

Two-stage stacked ensemble with domain-level interpretability:

- **Shared preprocessing** - identical data loading and feature engineering pipeline as the baselines notebook
- **Domain definition** - 87 features partitioned into six clinical subdomains: patient-related, surgery-related, hemodynamics, anesthetics, vasopressors, antihypertensives
- **Stage 1 (Base learners)** - domain-specific LightGBM classifiers trained with 5-fold patient-grouped CV to generate out-of-fold probability predictions per domain
- **Stage 2 (Meta-learner)** - L2 logistic regression trained on the stacked domain-level OOF predictions under the same grouped CV scheme
- **Domain contributions** - softmax-normalized meta-learner coefficients quantifying each domain's contribution to the final prediction
- **Evaluation** - AUROC, PR-AUC, Brier score, ROC curve, calibration curve, and decision curve analysis
- **Interpretability** - per-domain TreeSHAP analysis with global importance tables, beeswarm plots, and bar plots


## License

[MIT](LICENSE)
