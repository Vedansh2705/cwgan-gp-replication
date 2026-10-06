# baselines/smote.py
# =============================================================
# Step 7a: SMOTE Baseline (Paper Table 3)
# =============================================================
# SMOTE: minority class ke do samples ke beech ek "seedhi line"
# pe naya point banata hai (k-nearest neighbors use karke).
#
# Paper ke exact settings (Table 3):
#   k_neighbors = 5

from imblearn.over_sampling import SMOTE


def apply_smote(X_train, y_train, k_neighbors=5, random_state=None):
    """
    Training set par SMOTE oversampling apply karta hai.
    Sirf minority class ke naye samples banata hai jab tak
    dono classes balanced na ho jaayein (1:1).

    Args:
        X_train, y_train: training data (SIRF training set par,
                           test set kabhi nahi - data leakage yaad hai)
        k_neighbors: paper ke Table 3 ke hisaab se 5

    Returns:
        X_resampled, y_resampled: balanced training data
    """
    smote = SMOTE(k_neighbors=k_neighbors, random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

    from data.load_iris import load_imbalanced_iris
    from experiments.train_test_split_demo import get_train_test_splits

    X, y = load_imbalanced_iris()
    splits = get_train_test_splits(X, y, n_repeats=5, test_size=0.2)
    X_train, X_test, y_train, y_test = splits[0]

    print("SMOTE se PEHLE:")
    print(f"  Minority: {(y_train==1).sum()}, Majority: {(y_train==0).sum()}")

    X_resampled, y_resampled = apply_smote(X_train, y_train, k_neighbors=5, random_state=0)

    print("\nSMOTE ke BAAD:")
    print(f"  Minority: {(y_resampled==1).sum()}, Majority: {(y_resampled==0).sum()}")
    print(f"  Total samples: {len(y_resampled)} (pehle {len(y_train)} the)")