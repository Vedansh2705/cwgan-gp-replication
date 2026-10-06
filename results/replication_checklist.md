# Replication Quality Checklist (Phase 13)

Paper: Zheng et al., "Conditional Wasserstein GAN-Gradient Penalty-Based
Approach to Alleviating Imbalanced Data Classification" (Information
Sciences, 2019/2020)

Dataset used for this replication: **Iris (binary, versicolor=minority)**,
IR=2.00 — one of the paper's 15 UCI datasets.

Legend: MATCH | PARTIAL MATCH | DEVIATION | UNKNOWN

| # | Item | Status | Notes |
|---|------|--------|-------|
| 1 | Same dataset | MATCH | Iris, same class split as paper Table 1 (versicolor vs rest), verified IR=2.00 |
| 2 | Same preprocessing | DEVIATION | Paper specifies none. We added StandardScaler (mean=0, std=1) on training data before CWGAN-GP training — necessary for the generator to learn (raw-scale training diverged, see validation step). This is our documented assumption. |
| 3 | Same train/test split strategy | PARTIAL MATCH | 80/20 random split, 5 repeats — matches paper. We additionally used `stratify=y`, which the paper does not mention either way. |
| 4 | Same oversampling strategy | MATCH | Oversampling applied to training set only, after the split; test set never touched. Balanced to 1:1 (assumption — paper never states the oversampling target ratio explicitly). |
| 5 | Same architecture | DEVIATION | Paper gives only ranges (hidden units 32-256 for G, 16-128 for D; noise dim 25-100). We fixed: Generator 64→128 units, Critic 64→32 units, noise dim=32. Exact paper architecture is unrecoverable from the text (Phase 1, gap #1). |
| 6 | Same loss | MATCH | Implemented Wasserstein loss with gradient penalty exactly per Eq. 9-11. |
| 7 | Same gradient penalty | MATCH | lambda=10, interpolation per Eq. 8, penalty formula per Eq. 7. |
| 8 | Same optimizer | MATCH | Adam, alpha=0.0001, beta1=0, beta2=0.9 (paper's exact WGAN-GP/CWGAN-GP settings). |
| 9 | Same learning rate | MATCH | 0.0001, as specified. |
| 10 | Same batch size | MATCH | 32, as specified. |
| 11 | Same epochs | MATCH | 5000, as specified (after initial testing at 1000, which underperformed — see results). |
| 12 | Same critic iterations | MATCH | 5 critic updates per 1 generator update, as specified. |
| 13 | Same classifiers | MATCH | KNN, Logistic Regression, Random Forest, Decision Tree, SVM(RBF) — all scikit-learn defaults. SVM required `probability=True` (non-default) to compute AUC — this is a practical addition, not a paper deviation in intent. |
| 14 | Same metrics | MATCH | F-measure (beta=1), G-mean, AUC — formulas per Eq. 12-13 (Eq. 12 as printed has a typo; we used the standard F1 formula). |
| 15 | Same repetition count | MATCH | 5 repeats, results averaged (mean ± std), matching Section 3.3. |
| 16 | Same statistical testing | PARTIAL MATCH | Friedman + Nemenyi implemented per Section 3.4 (same alpha=0.05, same rank-based methodology). **Deviation:** paper uses N=15 datasets as blocks; we used N=5 repeats on a single dataset as blocks, since we replicated only 1 of 17 datasets. This gives much lower statistical power than the paper's design. |
| 17 | Same baseline methods | PARTIAL MATCH | Implemented: NONE (implicit in class balance), SMOTE, ADASYN, Borderline-SMOTE, ROS, CWGAN-GP (5 of paper's 9 methods). **Not yet implemented:** GAN, CGAN, WGAN-GP. |
| 18 | Same number of datasets | DEVIATION | Paper: 15 UCI + 2 real-world = 17 datasets. We used 1 (Iris) as an initial validation dataset, per the project's phased plan. |

## Summary

Of 18 checklist items: **10 MATCH, 4 PARTIAL MATCH, 4 DEVIATION, 0 UNKNOWN.**

The core CWGAN-GP training procedure (loss, gradient penalty, optimizer,
hyperparameters, epochs, critic ratio) replicates the paper's stated
settings exactly wherever the paper specifies them. The unavoidable
deviations are:
- **Architecture specifics** (paper gives ranges, not exact values — Phase 1, gap #1)
- **Preprocessing** (paper is silent; scaling was empirically necessary)
- **Scale of the study** (1 dataset vs 17; 5-repeat blocks vs 15-dataset blocks for statistics)
- **Baseline coverage** (5 of 9 methods implemented so far)

## Key empirical finding (on Iris only)

CWGAN-GP achieved the best mean rank among the 5 tested methods on
AUC (4.24/5) and tied for best on F-measure/G-mean, but Nemenyi tests
found **no statistically significant difference** between any pair of
methods at alpha=0.05, due to the small number of blocks (N=5). This
contrasts with the paper's result, where CWGAN-GP significantly
outperformed most baselines across all 15 UCI datasets. The most
likely explanation is that Iris is the simplest, least imbalanced
dataset in the paper's collection (IR=2.00, smallest of the 15), and
the paper itself notes (Section 4.4) that GAN-based methods show
their biggest advantage on more complex data distributions.

## Recommended next steps (not yet done)

1. Add a second, more complex/imbalanced dataset (e.g. Heart or German
   Credit) to increase statistical blocks and test whether CWGAN-GP's
   advantage grows with complexity, as the paper predicts.
2. Implement remaining baselines (GAN, CGAN, WGAN-GP) for the full
   9-method comparison.
3. Write up Phase 15 (research report) using this checklist as the
   "Limitations" and "Deviations from Original Study" sections.