# baselines/wgan_gp.py
# =============================================================
# Baseline: WGAN-GP (unconditional) - Paper Section 2.2
# =============================================================
# CWGAN-GP jaisa hi hai, BAS class label conditioning nahi hai.
# Generator sirf noise se banata hai (label nahi dikhaya jaata),
# Critic sirf sample dekhta hai (label ke bina).
#
# Isse hum test karte hain: conditioning (the "C" in CWGAN-GP)
# ka actually koi fayda hai ya nahi - paper ka Table 7 claim.

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model

from data.scaling import fit_scaler, scale_data, inverse_scale_data
from data.extract_minority import extract_minority


def build_unconditional_generator(noise_dim=32, n_features=4):
    """WGAN-GP Generator - koi label input nahi, sirf noise."""
    noise_input = layers.Input(shape=(noise_dim,), name="noise_input")
    x = layers.Dense(64, activation="relu")(noise_input)
    x = layers.Dense(128, activation="relu")(x)
    output = layers.Dense(n_features, activation=None, name="fake_sample")(x)
    return Model(inputs=noise_input, outputs=output, name="WGAN_Generator")


def build_unconditional_critic(n_features=4):
    """WGAN-GP Critic - koi label input nahi, sirf sample."""
    sample_input = layers.Input(shape=(n_features,), name="sample_input")
    x = layers.Dense(64, activation="relu")(sample_input)
    x = layers.Dense(32, activation="relu")(x)
    output = layers.Dense(1, activation=None, name="critic_score")(x)
    return Model(inputs=sample_input, outputs=output, name="WGAN_Critic")


def wgan_gradient_penalty(critic, real_samples, fake_samples):
    batch_size = real_samples.shape[0]
    eps = tf.random.uniform([batch_size, 1], 0.0, 1.0)
    interpolated = eps * real_samples + (1 - eps) * fake_samples

    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        pred = critic(interpolated, training=True)

    grads = tape.gradient(pred, interpolated)
    grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1) + 1e-12)
    gp = tf.reduce_mean((grad_norm - 1.0) ** 2)
    return gp


def wgan_critic_train_step(generator, critic, critic_optimizer,
                            real_samples, noise_dim, lambda_gp=10):
    batch_size = real_samples.shape[0]
    noise = tf.random.normal([batch_size, noise_dim])

    with tf.GradientTape() as tape:
        fake_samples = generator(noise, training=True)
        real_scores = critic(real_samples, training=True)
        fake_scores = critic(fake_samples, training=True)

        critic_loss = tf.reduce_mean(fake_scores) - tf.reduce_mean(real_scores)
        gp = wgan_gradient_penalty(critic, real_samples, fake_samples)
        critic_loss += lambda_gp * gp

    grads = tape.gradient(critic_loss, critic.trainable_variables)
    critic_optimizer.apply_gradients(zip(grads, critic.trainable_variables))
    return critic_loss


def wgan_generator_train_step(generator, critic, generator_optimizer,
                               batch_size, noise_dim):
    noise = tf.random.normal([batch_size, noise_dim])

    with tf.GradientTape() as tape:
        fake_samples = generator(noise, training=True)
        fake_scores = critic(fake_samples, training=True)
        gen_loss = -tf.reduce_mean(fake_scores)

    grads = tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(grads, generator.trainable_variables))
    return gen_loss


def train_wgan_gp(X_minority, noise_dim=32, n_features=4,
                   epochs=5000, batch_size=32, critic_steps=5,
                   lambda_gp=10, verbose_every=None):
    generator = build_unconditional_generator(noise_dim, n_features)
    critic = build_unconditional_critic(n_features)

    gen_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.0, beta_2=0.9)
    critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.0, beta_2=0.9)

    n_samples = X_minority.shape[0]

    for epoch in range(epochs):
        for _ in range(critic_steps):
            idx = np.random.randint(0, n_samples, batch_size)
            real_batch = tf.constant(X_minority[idx].astype(np.float32))
            c_loss = wgan_critic_train_step(
                generator, critic, critic_optimizer,
                real_batch, noise_dim, lambda_gp
            )

        g_loss = wgan_generator_train_step(
            generator, critic, gen_optimizer, batch_size, noise_dim
        )

        if verbose_every and (epoch + 1) % verbose_every == 0:
            print(f"Epoch {epoch+1}/{epochs} | Critic: {float(c_loss):.4f} | Gen: {float(g_loss):.4f}")

    return generator, critic


def apply_wgan_gp(X_train, y_train, epochs=5000, noise_dim=32,
                   minority_label=1, verbose_every=None):
    """
    SMOTE/CWGAN-GP jaisa hi interface - training data leta hai,
    minority class ke fake samples generate karke wapas balance
    kiya hua training set deta hai.

    IMPORTANT: Sirf minority data par train hota hai (label use
    nahi hota kahin bhi - ye hi WGAN-GP aur CWGAN-GP ka farak hai).
    """
    n_features = X_train.shape[1]

    scaler = fit_scaler(X_train)
    X_minority = extract_minority(X_train, y_train, minority_label=minority_label)
    X_minority_scaled = scale_data(scaler, X_minority)

    generator, critic = train_wgan_gp(
        X_minority_scaled, noise_dim=noise_dim, n_features=n_features,
        epochs=epochs, batch_size=32, critic_steps=5, lambda_gp=10,
        verbose_every=verbose_every
    )

    n_minority = (y_train == minority_label).sum()
    n_majority = len(y_train) - n_minority
    n_to_generate = n_majority - n_minority

    if n_to_generate <= 0:
        return X_train, y_train

    noise = tf.random.normal([n_to_generate, noise_dim])
    fake_samples_scaled = generator(noise, training=False).numpy()
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

    print("WGAN-GP se PEHLE:")
    print(f"  Minority: {(y_train==1).sum()}, Majority: {(y_train==0).sum()}")

    print("\nWGAN-GP training ho raha hai (1000 epochs, quick test)...")
    X_resampled, y_resampled = apply_wgan_gp(X_train, y_train, epochs=1000, verbose_every=250)

    print("\nWGAN-GP ke BAAD:")
    print(f"  Minority: {(y_resampled==1).sum()}, Majority: {(y_resampled==0).sum()}")
    print(f"  Total samples: {len(y_resampled)} (pehle {len(y_train)} the)")