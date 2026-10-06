# data/extract_minority.py
# =============================================================
# Step 4: Minority Samples Extract Karna (Paper Fig. 2)
# =============================================================
# Training set mein se sirf minority class (versicolor) ke
# samples nikaalte hain. CWGAN-GP ko sirf inhi samples se
# train karenge, taaki wo seekh sake minority class ka
# asli data-distribution kaisa hota hai.

import numpy as np


def extract_minority(X_train, y_train, minority_label=1):
    """
    Training set mein se sirf minority class ke samples nikaalta hai.

    Args:
        X_train: training features
        y_train: training labels
        minority_label: minority class ka label (default 1)

    Returns:
        X_minority: sirf minority class ke features
    """
    mask = (y_train == minority_label)
    X_minority = X_train[mask]
    return X_minority


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

    from data.load_iris import load_imbalanced_iris
    from experiments.train_test_split_demo import get_train_test_splits

    X, y = load_imbalanced_iris()
    splits = get_train_test_splits(X, y, n_repeats=5, test_size=0.2)

    # Sirf pehla split use karke dikhate hain (baaki 4 repeats mein bhi
    # yahi function chalega, jab full experiment loop banayenge)
    X_train, X_test, y_train, y_test = splits[0]

    X_minority = extract_minority(X_train, y_train, minority_label=1)

    print("Training set total samples:", len(X_train))
    print("Minority samples nikale gaye:", X_minority.shape)
    print()
    print("Pehle 3 minority samples:")
    print(X_minority[:3])