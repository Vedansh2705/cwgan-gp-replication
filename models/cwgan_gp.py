# models/cwgan_gp.py
# =============================================================
# Step 5c: CWGAN-GP Training Loop (Paper Section 2.3, Eq. 9-11)
# =============================================================
# Yahan hum Generator aur Critic ko ek saath train karte hain.
#
# Paper ke Loss Functions (Eq. 10, 11):
#   L(D) = -E[D(x_real|y)] + E[D(x_fake|y)] + lambda * GP
#   L(G) = -E[D(x_fake|y)]
#
# Paper ke hyperparameters (Section 3.2):
#   - lambda (gradient penalty coefficient) = 10
#   - Adam optimizer: alpha=0.0001, beta1=0, beta2=0.9
#   - Critic updates per Generator update = 5
#   - Batch size = 32, Epochs = 5000

import tensorflow as tf
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.generator import build_generator
from models.critic import build_critic


# =============================================================
# Part 1: Gradient Penalty calculate karna (Paper Eq. 7-8)
# =============================================================
def gradient_penalty(critic, real_samples, fake_samples, labels):
    """
    Real aur fake samples ke beech ek random point banate hain
    (interpolation), aur Critic ka gradient wahan check karte hain.
    Gradient ka norm 1 ke jitna paas ho, utna acha (Lipschitz constraint).

    Paper Eq. 8:  x_hat = eps*x_real + (1-eps)*x_fake
    """
    batch_size = real_samples.shape[0]

    # eps: har sample ke liye ek random number (0 se 1 ke beech)
    eps = tf.random.uniform([batch_size, 1], 0.0, 1.0)

    # Real aur fake ke beech ka interpolated point
    interpolated = eps * real_samples + (1 - eps) * fake_samples

    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        # Critic ka score is interpolated point par
        pred = critic([interpolated, labels], training=True)

    # Gradient nikaalo: score kitna badalta hai agar interpolated point
    # thoda sa idhar-udhar karein
    grads = tape.gradient(pred, interpolated)

    # Gradient ka norm (length) nikaalo
    grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1) + 1e-12)

    # Penalty: norm 1 se jitna door, utni zyada penalty
    gp = tf.reduce_mean((grad_norm - 1.0) ** 2)
    return gp


# =============================================================
# Part 2: Ek Critic training step
# =============================================================
def critic_train_step(generator, critic, critic_optimizer,
                       real_samples, labels, noise_dim, lambda_gp=10):
    """
    Critic ko ek baar train karta hai (Paper Eq. 10):
    L(D) = -E[D(x_real)] + E[D(x_fake)] + lambda * GP
    """
    batch_size = real_samples.shape[0]
    noise = tf.random.normal([batch_size, noise_dim])

    with tf.GradientTape() as tape:
        # Generator se nakli samples banwao
        fake_samples = generator([noise, labels], training=True)

        # Critic dono ko score de
        real_scores = critic([real_samples, labels], training=True)
        fake_scores = critic([fake_samples, labels], training=True)

        # Wasserstein loss ka core part
        critic_loss = tf.reduce_mean(fake_scores) - tf.reduce_mean(real_scores)

        # Gradient penalty add karo
        gp = gradient_penalty(critic, real_samples, fake_samples, labels)
        critic_loss += lambda_gp * gp

    # Sirf Critic ke weights update karo
    grads = tape.gradient(critic_loss, critic.trainable_variables)
    critic_optimizer.apply_gradients(zip(grads, critic.trainable_variables))

    return critic_loss


# =============================================================
# Part 3: Ek Generator training step
# =============================================================
def generator_train_step(generator, critic, generator_optimizer,
                          batch_size, labels, noise_dim):
    """
    Generator ko ek baar train karta hai (Paper Eq. 11):
    L(G) = -E[D(x_fake)]
    Generator chahta hai Critic fake samples ko HIGH score de.
    """
    noise = tf.random.normal([batch_size, noise_dim])

    with tf.GradientTape() as tape:
        fake_samples = generator([noise, labels], training=True)
        fake_scores = critic([fake_samples, labels], training=True)

        # Generator loss: negative of critic's score on fakes
        # (Generator chahta hai score zyada ho, isliye minimize karte
        #  hain -score, taaki score maximize ho)
        gen_loss = -tf.reduce_mean(fake_scores)

    grads = tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

    return gen_loss


# =============================================================
# Part 4: Poora training loop
# =============================================================
def train_cwgan_gp(X_minority, noise_dim=32, n_features=4, n_classes=2,
                    epochs=100, batch_size=32, critic_steps=5,
                    lambda_gp=10, verbose_every=10):
    """
    CWGAN-GP ko minority class data par train karta hai.

    NOTE: Paper 5000 epochs bolta hai. Hum abhi testing ke liye
    kam epochs (100) use kar rahe hain - Phase 6 mein hum ye
    dekhenge ki training theek chal rahi hai ya nahi, tabhi
    5000 tak badhayenge.
    """
    generator = build_generator(noise_dim, n_features, n_classes)
    critic = build_critic(n_features, n_classes)

    # Paper ke exact Adam settings (Section 3.2)
    gen_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.0, beta_2=0.9)
    critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.0, beta_2=0.9)

    # Minority class ka one-hot label sabke liye same rahega
    # (yahan class "1" = minority, one-hot = [0, 1])
    labels_np = np.zeros((batch_size, n_classes), dtype=np.float32)
    labels_np[:, 1] = 1
    labels = tf.constant(labels_np)   # numpy array ko TF tensor banaya

    history = {"critic_loss": [], "gen_loss": []}

    n_samples = X_minority.shape[0]

    for epoch in range(epochs):
        # ---- Critic ko 5 baar train karo ----
        for _ in range(critic_steps):
            idx = np.random.randint(0, n_samples, batch_size)
            real_batch = tf.constant(X_minority[idx].astype(np.float32))  # TF tensor

            c_loss = critic_train_step(
                generator, critic, critic_optimizer,
                real_batch, labels, noise_dim, lambda_gp
            )

        # ---- Generator ko 1 baar train karo ----
        g_loss = generator_train_step(
            generator, critic, gen_optimizer,
            batch_size, labels, noise_dim
        )

        history["critic_loss"].append(float(c_loss))
        history["gen_loss"].append(float(g_loss))

        if (epoch + 1) % verbose_every == 0:
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Critic Loss: {float(c_loss):.4f} | "
                  f"Gen Loss: {float(g_loss):.4f}")

    return generator, critic, history


if __name__ == "__main__":
    from data.load_iris import load_imbalanced_iris
    from experiments.train_test_split_demo import get_train_test_splits
    from data.extract_minority import extract_minority

    X, y = load_imbalanced_iris()
    splits = get_train_test_splits(X, y, n_repeats=5, test_size=0.2)
    X_train, X_test, y_train, y_test = splits[0]
    X_minority = extract_minority(X_train, y_train, minority_label=1)

    print(f"Training CWGAN-GP on {X_minority.shape[0]} minority samples...\n")

    generator, critic, history = train_cwgan_gp(
        X_minority,
        noise_dim=32,
        n_features=4,
        n_classes=2,
        epochs=100,       # abhi test ke liye kam rakha hai
        batch_size=32,
        critic_steps=5,
        lambda_gp=10,
        verbose_every=10
    )

    print("\nTraining complete!")