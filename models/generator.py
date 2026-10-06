# models/generator.py
# =============================================================
# Step 5a: CWGAN-GP Generator (Paper Section 2.3 + Section 3.2)
# =============================================================
# Generator ka kaam: noise + class label lekar ek nakli
# minority-class sample (4 features) generate karna.
#
# Paper se jo pata hai (Section 3.2):
#   - 4-layer neural network
#   - Noise dimension: 25-100 ke beech (hum 32 use kar rahe hain)
#   - Hidden layers: ReLU activation
#   - Hidden units: 32-256 ke beech
#   - No Dropout, no BatchNorm
#   - Conditional: class label bhi input mein jaata hai
#
# OUR ASSUMPTION (paper exact sizes nahi deta, sirf range deta hai):
#   Hidden Layer 1: 64 units
#   Hidden Layer 2: 128 units
#   (In dono ko hum aasani se badal sakte hain baad mein)

import tensorflow as tf
from tensorflow.keras import layers, Model


def build_generator(noise_dim=32, n_features=4, n_classes=2):
    """
    CWGAN-GP ka Generator banata hai.

    Args:
        noise_dim: kitne random numbers se shuru karenge (paper: 25-100)
        n_features: output mein kitne numbers chahiye
                    (Iris ke liye 4: sepal length/width, petal length/width)
        n_classes: kitni classes hain (Iris binary problem: 2)

    Returns:
        keras Model jo (noise, label) leta hai aur fake sample deta hai
    """
    # ---- Input 1: Random noise ----
    noise_input = layers.Input(shape=(noise_dim,), name="noise_input")

    # ---- Input 2: Class label (conditioning) ----
    # Hum label ko one-hot encode karke denge (e.g. class 1 -> [0, 1])
    label_input = layers.Input(shape=(n_classes,), name="label_input")

    # ---- Concatenate: noise aur label ko jod do ----
    # Yahi woh step hai jo GAN ko "Conditional" banata hai (Section 2.3)
    x = layers.Concatenate()([noise_input, label_input])

    # ---- Hidden Layer 1 ----
    x = layers.Dense(64, activation="relu")(x)

    # ---- Hidden Layer 2 ----
    x = layers.Dense(128, activation="relu")(x)

    # ---- Output Layer ----
    # No activation (linear) - paper WGAN-GP/CWGAN-GP output mein
    # koi activation function use nahi karta (Section 3.2)
    output = layers.Dense(n_features, activation=None, name="fake_sample")(x)

    model = Model(inputs=[noise_input, label_input], outputs=output, name="Generator")
    return model


if __name__ == "__main__":
    generator = build_generator(noise_dim=32, n_features=4, n_classes=2)
    generator.summary()

    # Test: ek chhota batch generate karke dekhte hain
    import numpy as np

    batch_size = 5
    noise = np.random.normal(0, 1, size=(batch_size, 32))
    labels = np.zeros((batch_size, 2))
    labels[:, 1] = 1  # sabko class "1" (minority) bata rahe hain

    fake_samples = generator([noise, labels])
    print("\nGenerated fake samples shape:", fake_samples.shape)
    print("Pehle 2 fake samples (abhi random hain, training se pehle):")
    print(fake_samples.numpy()[:2])