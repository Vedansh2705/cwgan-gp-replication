# baselines/cwgan_gp_oversample.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import tensorflow as tf

from data.scaling import fit_scaler, scale_data, inverse_scale_data
from data.extract_minority import extract_minority
from models.cwgan_gp import train_cwgan_gp


def generate_fake_samples(generator, n_samples, noise_dim=32, n_classes=2, minority_class_idx=1):
    noise = tf.random.normal([n_samples, noise_dim])
    labels = np.zeros((n_samples, n_classes), dtype=np.float32)
    labels[:, minority_class_idx] = 1
    labels = tf.constant(labels)
    fake_samples = generator([noise, labels], training=False)
    return fake_samples.numpy()


def apply_cwgan_gp(X_train, y_train, epochs=1000, noise_dim=32,
                    n_classes=2, minority_label=1, verbose_every=None):
    n_features = X_train.shape[1]

    scaler = fit_scaler(X_train)
    X_minority = extract_minority(X_train, y_train, minority_label=minority_label)
    X_minority_scaled = scale_data(scaler, X_minority)

    generator, critic, history = train_cwgan_gp(
        X_minority_scaled,
        noise_dim=noise_dim,
        n_features=n_features,
        n_classes=n_classes,
        epochs=epochs,
        batch_size=32,
        critic_steps=5,
        lambda_gp=10,
        verbose_every=verbose_every if verbose_every else epochs + 1
    )

    n_minority = (y_train == minority_label).sum()
    n_majority = len(y_train) - n_minority
    n_to_generate = n_majority - n_minority

    if n_to_generate <= 0:
        return X_train, y_train

    fake_samples_scaled = generate_fake_samples(
        generator, n_samples=n_to_generate,
        noise_dim=noise_dim, n_classes=n_classes,
        minority_class_idx=minority_label
    )
    fake_samples = inverse_scale_data(scaler, fake_samples_scaled)

    fake_labels = np.full(n_to_generate, minority_label)
    X_resampled = np.vstack([X_train, fake_samples])
    y_resampled = np.concatenate([y_train, fake_labels])

    return X_resampled, y_resampled


if __name__ == "__main__":
    from data.load_iris import load_imbalanced_iris
    from experiments.train_test_split_demo import get_train_test_splits

    X, y = load_imbalanced_iris()
    splits = get_train_test_splits(X, y, n_repeats=5, test_size=0.2)
    X_train, X_test, y_train, y_test = splits[0]

    print("CWGAN-GP se PEHLE:")
    print(f"  Minority: {(y_train==1).sum()}, Majority: {(y_train==0).sum()}")

    print("\nCWGAN-GP training ho raha hai (1000 epochs, thoda time lagega)...")
    X_resampled, y_resampled = apply_cwgan_gp(X_train, y_train, epochs=1000)

    print("\nCWGAN-GP ke BAAD:")
    print(f"  Minority: {(y_resampled==1).sum()}, Majority: {(y_resampled==0).sum()}")
    print(f"  Total samples: {len(y_resampled)} (pehle {len(y_train)} the)")