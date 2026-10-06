# baselines/gan_and_cgan.py
# =============================================================
# Baselines: GAN aur CGAN (Paper Section 2.1, Eq. 1-2)
# =============================================================
# GAN/CGAN, WGAN-GP/CWGAN-GP se alag hain:
#   - Binary cross-entropy loss (Wasserstein nahi)
#   - Discriminator sigmoid output deta hai (0-1 probability)
#   - No gradient penalty
#   - 1 discriminator update per generator update (5 nahi)
#   - Adam DEFAULT settings (beta1=0 nahi, standard 0.9)
#
# CGAN = GAN + class label conditioning (jaise CWGAN-GP WGAN-GP se hai)

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model

from data.scaling import fit_scaler, scale_data, inverse_scale_data
from data.extract_minority import extract_minority


# =============================================================
# Part 1: Models (unconditional GAN aur conditional CGAN)
# =============================================================
def build_gan_generator(noise_dim=32, n_features=4):
    """GAN Generator - no label."""
    noise_input = layers.Input(shape=(noise_dim,))
    x = layers.Dense(64, activation="relu")(noise_input)
    x = layers.Dense(128, activation="relu")(x)
    output = layers.Dense(n_features, activation=None)(x)
    return Model(inputs=noise_input, outputs=output, name="GAN_Generator")


def build_gan_discriminator(n_features=4):
    """GAN Discriminator - sigmoid output (probability), no label."""
    sample_input = layers.Input(shape=(n_features,))
    x = layers.Dense(64, activation="relu")(sample_input)
    x = layers.Dense(32, activation="relu")(x)
    output = layers.Dense(1, activation="sigmoid")(x)  # <-- sigmoid!
    return Model(inputs=sample_input, outputs=output, name="GAN_Discriminator")


def build_cgan_generator(noise_dim=32, n_features=4, n_classes=2):
    """CGAN Generator - label conditioning hai."""
    noise_input = layers.Input(shape=(noise_dim,))
    label_input = layers.Input(shape=(n_classes,))
    x = layers.Concatenate()([noise_input, label_input])
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)
    output = layers.Dense(n_features, activation=None)(x)
    return Model(inputs=[noise_input, label_input], outputs=output, name="CGAN_Generator")


def build_cgan_discriminator(n_features=4, n_classes=2):
    """CGAN Discriminator - sigmoid output, label conditioning hai."""
    sample_input = layers.Input(shape=(n_features,))
    label_input = layers.Input(shape=(n_classes,))
    x = layers.Concatenate()([sample_input, label_input])
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)
    output = layers.Dense(1, activation="sigmoid")(x)
    return Model(inputs=[sample_input, label_input], outputs=output, name="CGAN_Discriminator")


# =============================================================
# Part 2: Training - GAN (unconditional, BCE loss)
# =============================================================
bce = tf.keras.losses.BinaryCrossentropy()


def train_gan(X_minority, noise_dim=32, n_features=4, epochs=5000,
              batch_size=32, verbose_every=None):
    generator = build_gan_generator(noise_dim, n_features)
    discriminator = build_gan_discriminator(n_features)

    # Paper: Adam DEFAULT settings (standard TF defaults)
    gen_optimizer = tf.keras.optimizers.Adam()
    disc_optimizer = tf.keras.optimizers.Adam()

    n_samples = X_minority.shape[0]

    for epoch in range(epochs):
        idx = np.random.randint(0, n_samples, batch_size)
        real_batch = tf.constant(X_minority[idx].astype(np.float32))
        noise = tf.random.normal([batch_size, noise_dim])

        # ---- Discriminator step (1 baar, GAN/CGAN mein) ----
        with tf.GradientTape() as tape:
            fake_batch = generator(noise, training=True)
            real_pred = discriminator(real_batch, training=True)
            fake_pred = discriminator(fake_batch, training=True)

            real_loss = bce(tf.ones_like(real_pred), real_pred)
            fake_loss = bce(tf.zeros_like(fake_pred), fake_pred)
            disc_loss = real_loss + fake_loss

        grads = tape.gradient(disc_loss, discriminator.trainable_variables)
        disc_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

        # ---- Generator step ----
        noise = tf.random.normal([batch_size, noise_dim])
        with tf.GradientTape() as tape:
            fake_batch = generator(noise, training=True)
            fake_pred = discriminator(fake_batch, training=True)
            # Generator chahta hai discriminator fake ko "real" (1) samjhe
            gen_loss = bce(tf.ones_like(fake_pred), fake_pred)

        grads = tape.gradient(gen_loss, generator.trainable_variables)
        gen_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

        if verbose_every and (epoch + 1) % verbose_every == 0:
            print(f"  [GAN] Epoch {epoch+1}/{epochs} | D: {float(disc_loss):.4f} | G: {float(gen_loss):.4f}")

    return generator


# =============================================================
# Part 3: Training - CGAN (conditional, BCE loss)
# =============================================================
def train_cgan(X_minority, noise_dim=32, n_features=4, n_classes=2,
               epochs=5000, batch_size=32, minority_label=1, verbose_every=None):
    generator = build_cgan_generator(noise_dim, n_features, n_classes)
    discriminator = build_cgan_discriminator(n_features, n_classes)

    gen_optimizer = tf.keras.optimizers.Adam()
    disc_optimizer = tf.keras.optimizers.Adam()

    labels_np = np.zeros((batch_size, n_classes), dtype=np.float32)
    labels_np[:, minority_label] = 1
    labels = tf.constant(labels_np)

    n_samples = X_minority.shape[0]

    for epoch in range(epochs):
        idx = np.random.randint(0, n_samples, batch_size)
        real_batch = tf.constant(X_minority[idx].astype(np.float32))
        noise = tf.random.normal([batch_size, noise_dim])

        with tf.GradientTape() as tape:
            fake_batch = generator([noise, labels], training=True)
            real_pred = discriminator([real_batch, labels], training=True)
            fake_pred = discriminator([fake_batch, labels], training=True)

            real_loss = bce(tf.ones_like(real_pred), real_pred)
            fake_loss = bce(tf.zeros_like(fake_pred), fake_pred)
            disc_loss = real_loss + fake_loss

        grads = tape.gradient(disc_loss, discriminator.trainable_variables)
        disc_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

        noise = tf.random.normal([batch_size, noise_dim])
        with tf.GradientTape() as tape:
            fake_batch = generator([noise, labels], training=True)
            fake_pred = discriminator([fake_batch, labels], training=True)
            gen_loss = bce(tf.ones_like(fake_pred), fake_pred)

        grads = tape.gradient(gen_loss, generator.trainable_variables)
        gen_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

        if verbose_every and (epoch + 1) % verbose_every == 0:
            print(f"  [CGAN] Epoch {epoch+1}/{epochs} | D: {float(disc_loss):.4f} | G: {float(gen_loss):.4f}")

    return generator


# =============================================================
# Part 4: SMOTE-jaisa interface - apply_gan() aur apply_cgan()
# =============================================================
def apply_gan(X_train, y_train, epochs=5000, noise_dim=32,
              minority_label=1, verbose_every=None):
    n_features = X_train.shape[1]

    scaler = fit_scaler(X_train)
    X_minority = extract_minority(X_train, y_train, minority_label=minority_label)
    X_minority_scaled = scale_data(scaler, X_minority)

    generator = train_gan(
        X_minority_scaled, noise_dim=noise_dim, n_features=n_features,
        epochs=epochs, batch_size=32, verbose_every=verbose_every
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


def apply_cgan(X_train, y_train, epochs=5000, noise_dim=32, n_classes=2,
               minority_label=1, verbose_every=None):
    n_features = X_train.shape[1]

    scaler = fit_scaler(X_train)
    X_minority = extract_minority(X_train, y_train, minority_label=minority_label)
    X_minority_scaled = scale_data(scaler, X_minority)

    generator = train_cgan(
        X_minority_scaled, noise_dim=noise_dim, n_features=n_features,
        n_classes=n_classes, epochs=epochs, batch_size=32,
        minority_label=minority_label, verbose_every=verbose_every
    )

    n_minority = (y_train == minority_label).sum()
    n_majority = len(y_train) - n_minority
    n_to_generate = n_majority - n_minority

    if n_to_generate <= 0:
        return X_train, y_train

    noise = tf.random.normal([n_to_generate, noise_dim])
    gen_labels_np = np.zeros((n_to_generate, n_classes), dtype=np.float32)
    gen_labels_np[:, minority_label] = 1
    gen_labels = tf.constant(gen_labels_np)

    fake_samples_scaled = generator([noise, gen_labels], training=False).numpy()
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

    print("Testing GAN (1000 epochs, quick check)...")
    X_gan, y_gan = apply_gan(X_train, y_train, epochs=1000, verbose_every=250)
    print(f"  Result: Minority={(y_gan==1).sum()}, Majority={(y_gan==0).sum()}, Total={len(y_gan)}")

    print("\nTesting CGAN (1000 epochs, quick check)...")
    X_cgan, y_cgan = apply_cgan(X_train, y_train, epochs=1000, verbose_every=250)
    print(f"  Result: Minority={(y_cgan==1).sum()}, Majority={(y_cgan==0).sum()}, Total={len(y_cgan)}")