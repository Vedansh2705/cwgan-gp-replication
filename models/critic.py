# models/critic.py
# =============================================================
# Step 5b: CWGAN-GP Critic / Discriminator (Paper Section 2.3 + 3.2)
# =============================================================
# Critic ka kaam: ek sample (asli ya nakli) lekar ek SCORE dena
# - jitna zyada score, utna zyada "asli" lagta hai use.
#
# Paper se jo pata hai (Section 3.2):
#   - 4-layer neural network
#   - Hidden units: 16-128 ke beech (Generator se alag range)
#   - ReLU activation, no Dropout, no BatchNorm
#   - Conditional: class label bhi input mein jaata hai
#   - IMPORTANT: Output layer mein KOI activation nahi (linear)
#     GAN/CGAN ka Discriminator sigmoid use karta hai (0-1 probability),
#     lekin WGAN-GP/CWGAN-GP ka Critic koi bhi real number de sakta hai.
#
# OUR ASSUMPTION (paper exact sizes nahi deta):
#   Hidden Layer 1: 64 units
#   Hidden Layer 2: 32 units

import tensorflow as tf
from tensorflow.keras import layers, Model


def build_critic(n_features=4, n_classes=2):
    """
    CWGAN-GP ka Critic banata hai.

    Args:
        n_features: input sample mein kitne numbers hain (Iris: 4)
        n_classes: kitni classes hain (Iris binary: 2)

    Returns:
        keras Model jo (sample, label) leta hai aur ek score deta hai
    """
    # ---- Input 1: Sample (asli ya nakli, dono yahi shape ke honge) ----
    sample_input = layers.Input(shape=(n_features,), name="sample_input")

    # ---- Input 2: Class label (conditioning) ----
    label_input = layers.Input(shape=(n_classes,), name="label_input")

    # ---- Concatenate: sample aur label ko jod do ----
    x = layers.Concatenate()([sample_input, label_input])

    # ---- Hidden Layer 1 ----
    x = layers.Dense(64, activation="relu")(x)

    # ---- Hidden Layer 2 ----
    x = layers.Dense(32, activation="relu")(x)

    # ---- Output Layer ----
    # Sirf 1 number (score), koi activation nahi (linear)
    output = layers.Dense(1, activation=None, name="critic_score")(x)

    model = Model(inputs=[sample_input, label_input], outputs=output, name="Critic")
    return model


if __name__ == "__main__":
    critic = build_critic(n_features=4, n_classes=2)
    critic.summary()

    # Test: kuch fake "samples" bana kar critic ko dikhate hain
    import numpy as np

    batch_size = 5
    fake_samples = np.random.normal(0, 1, size=(batch_size, 4))  # dummy data
    labels = np.zeros((batch_size, 2))
    labels[:, 1] = 1  # class "1" (minority)

    scores = critic([fake_samples, labels])
    print("\nCritic scores shape:", scores.shape)
    print("Scores (abhi random hain, training se pehle):")
    print(scores.numpy())   