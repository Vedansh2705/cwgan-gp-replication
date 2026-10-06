# evaluation/validate_generator.py
# =============================================================
# Step 6: CWGAN-GP Validation (Paper Section: "diagnostic outputs")
# =============================================================
# Ab hum data ko SCALE karke train karenge (Generator ke liye
# seekhna aasan banane ke liye), aur fake samples ko wapas
# ORIGINAL scale mein convert karke real data se compare karenge.

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from data.load_iris import load_imbalanced_iris
from experiments.train_test_split_demo import get_train_test_splits
from data.extract_minority import extract_minority
from data.scaling import fit_scaler, scale_data, inverse_scale_data
from models.cwgan_gp import train_cwgan_gp


def generate_fake_samples(generator, n_samples, noise_dim=32, n_classes=2, minority_class_idx=1):
    noise = tf.random.normal([n_samples, noise_dim])
    labels = np.zeros((n_samples, n_classes), dtype=np.float32)
    labels[:, minority_class_idx] = 1
    labels = tf.constant(labels)

    fake_samples = generator([noise, labels], training=False)
    return fake_samples.numpy()


if __name__ == "__main__":
    # ---- Data taiyaar karo ----
    X, y = load_imbalanced_iris()
    splits = get_train_test_splits(X, y, n_repeats=5, test_size=0.2)
    X_train, X_test, y_train, y_test = splits[0]
    X_minority = extract_minority(X_train, y_train, minority_label=1)

    # ---- Scaler fit karo poore training set par,
    #      phir minority data ko scale karo ----
    scaler = fit_scaler(X_train)
    X_minority_scaled = scale_data(scaler, X_minority)

    print("Scaling se pehle - minority mean:", X_minority.mean(axis=0).round(2))
    print("Scaling ke baad   - minority mean:", X_minority_scaled.mean(axis=0).round(2))
    print()

    print(f"Training CWGAN-GP on {X_minority_scaled.shape[0]} SCALED minority samples...\n")

    # ---- SCALED data par train karo, 1000 epochs ----
    generator, critic, history = train_cwgan_gp(
        X_minority_scaled,
        noise_dim=32,
        n_features=4,
        n_classes=2,
        epochs=1000,
        batch_size=32,
        critic_steps=5,
        lambda_gp=10,
        verbose_every=200
    )

    print("\nTraining complete! Ab validation kar rahe hain...\n")

    # ---- Fake samples generate karo (SCALED space mein) ----
    fake_samples_scaled = generate_fake_samples(
        generator, n_samples=X_minority.shape[0],
        noise_dim=32, n_classes=2, minority_class_idx=1
    )

    # ---- Fake samples ko wapas ORIGINAL scale mein le aao ----
    fake_samples = inverse_scale_data(scaler, fake_samples_scaled)

    # ---- 1. Numbers se compare karo (original scale mein) ----
    feature_names = ["sepal length", "sepal width", "petal length", "petal width"]

    print("=" * 60)
    print("REAL minority samples - Mean aur Std (per feature):")
    print("=" * 60)
    for i, name in enumerate(feature_names):
        print(f"  {name:15s}: mean={X_minority[:, i].mean():.3f}, std={X_minority[:, i].std():.3f}")

    print()
    print("=" * 60)
    print("FAKE (generated) samples - Mean aur Std (per feature):")
    print("=" * 60)
    for i, name in enumerate(feature_names):
        print(f"  {name:15s}: mean={fake_samples[:, i].mean():.3f}, std={fake_samples[:, i].std():.3f}")

    # ---- 2. PCA Visualization (original scale data par) ----
    combined = np.vstack([X_minority, fake_samples])
    pca = PCA(n_components=2)
    combined_2d = pca.fit_transform(combined)

    n_real = X_minority.shape[0]
    real_2d = combined_2d[:n_real]
    fake_2d = combined_2d[n_real:]

    plt.figure(figsize=(8, 6))
    plt.scatter(real_2d[:, 0], real_2d[:, 1], c='blue', label='Real minority', alpha=0.6, s=60)
    plt.scatter(fake_2d[:, 0], fake_2d[:, 1], c='red', label='Fake (generated)', alpha=0.6, s=60, marker='x')
    plt.title("PCA: Real vs Fake Minority Samples (Iris versicolor) - Scaled Training")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.grid(True, alpha=0.3)

    output_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'pca_real_vs_fake.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPCA plot saved: {output_path}")
    plt.show()