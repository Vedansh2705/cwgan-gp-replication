# CWGAN-GP Replication Project

ATML college project replicating:
"Conditional Wasserstein Generative Adversarial Network-Gradient Penalty-Based
Approach to Alleviating Imbalanced Data Classification"
(Zheng et al., Information Sciences, 2019/2020, DOI: 10.1016/j.ins.2019.10.014)

## Status
- [x] Phase 1: Paper analysis
- [ ] Phase 2: Replication specification
- [x] Phase 3: First dataset selected (Iris, IR=2.00)
- [ ] Phase 5: CWGAN-GP implementation
- [ ] Phase 6: Validation
- [ ] Phase 7: Baselines
- [ ] Phase 8: Classifiers
- [ ] Phase 9-12: Experiments, metrics, statistics, results
- [ ] Phase 15-16: Report, presentation

## Folder structure
- `data/` - dataset loading + preprocessing
- `models/` - CWGAN-GP generator, critic, training loop
- `baselines/` - SMOTE, ADASYN, ROS, GAN, CGAN, WGAN-GP
- `classifiers/` - KNN, LR, RF, DT, SVM wrappers
- `evaluation/` - F-measure, G-mean, AUC implementations
- `experiments/` - scripts that run the full pipeline
- `results/` - output tables/plots (generated, not committed)
- `notebooks/` - exploratory / Colab notebooks

## Setup
```bash
pip install -r requirements.txt
```

## Run Step 1 (data loading)
```bash
python data/load_iris.py
```
