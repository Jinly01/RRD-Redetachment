Interpretable Machine Learning Prediction of Retinal Re-detachment After Primary Pars Plana Vitrectomy with Silicone Oil Tamponade for Rhegmatogenous Retinal Detachment: A Dual-Center Development and External Validation Study
Overview
This repository contains the complete source code for developing, validating, and visualizing an interpretable XGBoost prediction model for retinal re-detachment within 12 months after primary pars plana vitrectomy (PPV) with silicone oil (SO) tamponade in patients with rhegmatogenous retinal detachment (RRD). The study is a retrospective dual-center cohort study conducted at two tertiary ophthalmic teaching hospitals in China (development cohort: n = 701; external validation cohort: n = 308).
Repository Structure
├── Model.py            # Core model development and evaluation pipeline
├── SA1.py              # Sensitivity analysis 1: retinal break phenotype clustering & survival analysis
├── SA2.py              # Sensitivity analysis 2: incremental value of intraoperative variables
├── Visualization.py    # Publication-quality figure generation
└── README.md
File Descriptions
Model.py — Model Development & Evaluation
End-to-end pipeline covering data preprocessing through final model export.

Data preprocessing: type-aware column transformers with log-transformation for skewed features, iterative (MICE) imputation for continuous variables, mode imputation for categorical variables, and standardization.
Feature selection: embedded LASSO (SelectFromModel with L1-penalized logistic regression) applied within each cross-validation fold; selection frequency tracked across all folds.
Candidate algorithms: Ridge Logistic Regression, Random Forest, SVM-RBF, XGBoost, and Stacking Ensemble (with diversity-aware base learner selection).
Nested cross-validation: outer loop 5-fold × 20 repeats generating unbiased out-of-fold (OOF) predictions; inner loop 5-fold for hyperparameter tuning (optimizing AUPRC).
Hierarchical model selection: pre-specified strategy incorporating discrimination (AUROC, AUPRC), calibration (Brier score, calibration slope), and imbalanced-data metrics (MCC, G-mean).
Comprehensive evaluation: bootstrap 95% CIs for all metrics, calibration analysis (slope and intercept), Youden's index threshold optimization, and DeLong test data export.

SA1.py — Sensitivity Analysis 1: Break Phenotype Clustering
Unsupervised phenotyping of retinal break characteristics with downstream impact assessment.

K-Prototypes clustering for mixed-type break features (continuous + categorical) with bootstrap stability evaluation.
Cluster validation: silhouette score, Calinski-Harabasz index, adjusted Rand index, and cross-cohort reproducibility.
Sensitivity model: replaces original break variables with cluster labels; evaluates change in discrimination via NRI/IDI.
Survival analysis: Kaplan-Meier curves with multivariate log-rank tests stratified by cluster membership.

SA2.py — Sensitivity Analysis 2: Intraoperative Variable Assessment
Quantifies whether adding intraoperative variables (surgery duration, PFCL use, phacovitrectomy) improves prediction beyond the preoperative-only model.

Trains an augmented XGBoost model (preoperative + intraoperative features) using identical nested CV protocol.
Computes paired AUROC/AUPRC differences with bootstrap CIs, NRI, and IDI.
Generates comparative ROC, Precision-Recall, and calibration plots (preoperative vs. augmented).

Visualization.py — Publication Figures
Generates all manuscript and supplementary figures at 600 DPI with a consistent minimalist SCI style.

ROC and Precision-Recall curves (internal + external, multi-model comparison).
Calibration plots with LOWESS smoothing and Hosmer-Lemeshow decile overlays.
Decision curve analysis (DCA) with net benefit across threshold probabilities.
SHAP summary plots (beeswarm), SHAP dependence plots, and SHAP waterfall plots for individual predictions.
Three-tier risk stratification bar charts with observed event rates.
Feature selection frequency heatmaps.
DeLong test pairwise comparison matrices.
Competing risk subgroup analysis visualizations.

Requirements
python >= 3.9
numpy
pandas
scikit-learn
xgboost
imbalanced-learn
shap
matplotlib
seaborn
scipy
joblib
lifelines
kmodes
Install all dependencies:
bashpip install numpy pandas scikit-learn xgboost imbalanced-learn shap matplotlib seaborn scipy joblib lifelines kmodes
Usage
1. Model Development
bashpython Model.py
This trains all candidate models via nested cross-validation, selects the optimal algorithm, and exports a model package (.pkl) along with OOF predictions to model_results/.
2. Sensitivity Analyses
bashpython SA1.py    # Break phenotype clustering & survival analysis
python SA2.py    # Intraoperative variable incremental value assessment
3. Figure Generation
bashpython Visualization.py
Outputs all publication-quality figures to figures/.
Data Availability
Due to patient privacy and institutional review board restrictions, individual-level clinical data are not publicly available. De-identified data may be made available from the corresponding author upon reasonable request and institutional approval.
License
Copyright © 2026. All rights reserved. This code is provided for academic review and reproducibility purposes only. Redistribution or commercial use requires prior written consent from the authors.
