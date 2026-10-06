# data/load_lymphography.py
# =============================================================
# Dataset 5: Lymphography (Paper Table 1, ID 3)
# =============================================================

from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np


def load_imbalanced_lymphography():
    lymph = fetch_ucirepo(id=63)

    X_df = lymph.data.features
    y_df = lymph.data.targets

    print(f"  [DEBUG] X_df shape (raw): {X_df.shape}")
    print(f"  [DEBUG] y_df shape (raw): {y_df.shape}")
    print(f"  [DEBUG] X_df columns: {list(X_df.columns)}")
    print(f"  [DEBUG] y_df columns: {list(y_df.columns)}")
    print(f"  [DEBUG] X_df NaN count per column:\n{X_df.isna().sum()}")
    print(f"  [DEBUG] y_df NaN count: {y_df.isna().sum().values}")

    # Concat se pehle index reset karo (kabhi mismatch index ki
    # wajah se rows drop ho jaate hain)
    X_df = X_df.reset_index(drop=True)
    y_df = y_df.reset_index(drop=True)

    X = X_df.values.astype(np.float64)
    raw_target = y_df.values.ravel()

    print(f"  [DEBUG] Target unique values: {np.unique(raw_target)}")
    print(f"  [DEBUG] X shape (final): {X.shape}")

    y = (raw_target == 3).astype(int)
    return X, y


if __name__ == "__main__":
    X, y = load_imbalanced_lymphography()
    n_minority = (y == 1).sum()
    n_majority = (y == 0).sum()

    print("\nTotal samples:", len(y))
    print("Feature count:", X.shape[1], "(paper says 18)")
    print(f"Minority: {n_minority} (paper says 61)")
    print(f"Majority: {n_majority} (paper says 87)")
    if n_minority > 0:
        IR = n_majority / n_minority
        print(f"IR: {IR:.2f}   (paper's Table 1 value = 1.43)")