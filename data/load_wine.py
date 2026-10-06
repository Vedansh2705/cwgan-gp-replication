# data/load_wine.py
# =============================================================
# Dataset 6: Wine (Paper Table 1, ID 5)
# =============================================================
# Paper ke Table 1 ke hisaab se:
#   Minority class = 1, 59 samples
#   Majority class = all other, 119 samples
#   IR = 2.02
#   Features = 13

from sklearn.datasets import load_wine
import numpy as np


def load_imbalanced_wine():
    """
    sklearn ka built-in Wine dataset (UCI se hi hai) - koi
    network fetch ki zaroorat nahi, isliye fast aur reliable.
    """
    wine = load_wine()
    X = wine.data.astype(np.float64)
    raw_target = wine.target  # 0, 1, 2 - sklearn 0-indexed hai

    # sklearn class 0 = UCI class 1 (59 samples) -> minority
    # baaki sab (sklearn 1, 2) = majority
    y = (raw_target == 0).astype(int)

    return X, y


if __name__ == "__main__":
    X, y = load_imbalanced_wine()
    n_minority = (y == 1).sum()
    n_majority = (y == 0).sum()
    IR = n_majority / n_minority

    print("Total samples:", len(y))
    print("Feature count:", X.shape[1], "(paper says 13)")
    print(f"Minority: {n_minority} (paper says 59)")
    print(f"Majority: {n_majority} (paper says 119)")
    print(f"IR: {IR:.2f}   (paper's Table 1 value = 2.02)")