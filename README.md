# CWGAN-GP Replication Project

## Conditional Wasserstein GAN with Gradient Penalty for Imbalanced Data Classification

This project is a partial replication of the research paper:

> **"Conditional Wasserstein Generative Adversarial Network-Gradient Penalty-Based Approach to Alleviating Imbalanced Data Classification"**

**Authors:** Zheng et al.  
**Journal:** Information Sciences  
**DOI:** [10.1016/j.ins.2019.10.014](https://doi.org/10.1016/j.ins.2019.10.014)

---

## 📌 Project Overview

Imbalanced classification occurs when one class contains significantly fewer samples than another. In such cases, machine learning classifiers can become biased toward the majority class.

This project replicates the **CWGAN-GP** approach proposed by Zheng et al. to generate realistic synthetic minority-class samples and improve classification performance on imbalanced datasets.

The project compares CWGAN-GP with traditional oversampling methods and other generative approaches.

### Main Idea

```text
Imbalanced Dataset
       ↓
Identify Minority Class
       ↓
Train Generative Model
       ↓
Generate Synthetic Minority Samples
       ↓
Balance Training Dataset
       ↓
Train Classifiers
       ↓
Evaluate Performance
