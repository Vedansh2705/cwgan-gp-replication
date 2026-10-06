# data/load_haberman.py
# =============================================================
# Dataset 3: Haberman's Survival (Paper Table 1, ID 8)
# =============================================================
# Paper ke Table 1 ke hisaab se:
#   Minority class = 2 (died within 5 years), 81 samples
#   Majority class = 1 (survived 5+ years), 225 samples
#   IR = 2.78
#   Features = 3 (age, year of operation, positive axillary nodes)

from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np


def load_imbalanced_haberman():
    """
    UCI Haberman's Survival dataset (ID=43) ko fetch karke,
    paper ke Table 1 ke hisaab se binary imbalanced dataset banata hai.
    """
    haberman = fetch_ucirepo(id=43)

    X_df = haberman.data.features
    y_df = haberman.data.targets

    combined = pd.concat([X_df, y_df], axis=1).dropna()
    X_df = combined[X_df.columns]
    y_df = combined[y_df.columns]

    X = X_df.values.astype(np.float64)

    # UCI target: 1 = survived (majority), 2 = died (minority)
    raw_target = y_df.values.ravel()
    y = (raw_target == 2).astype(int)  # 1=minority(died), 0=majority

    return X, y


if __name__ == "__main__":
    X, y = load_imbalanced_haberman()
    n_minority = (y == 1).sum()
    n_majority = (y == 0).sum()
    IR = n_majority / n_minority

    print("Total samples:", len(y))
    print("Feature count:", X.shape[1], "(paper says 3)")
    print("Minority samples:", n_minority)
    print("Majority samples:", n_majority)
    print(f"Imbalance Ratio (IR): {IR:.2f}   (paper's Table 1 value = 2.78)")