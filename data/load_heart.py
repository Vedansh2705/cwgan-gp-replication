# data/load_heart.py
# =============================================================
# Dataset 2: Heart (Paper Table 1, ID 1)
# =============================================================
# Paper ke Table 1 ke hisaab se:
#   Minority class = 2 (presence of heart disease), 120 samples
#   Majority class = 1 (absence), 150 samples
#   IR = 1.25
#   Features = 12 (paper mein)
#
# IMPORTANT DEVIATION (Phase 1 se pehle hi note kiya tha):
# UCI/Statlog Heart dataset mein asal mein 13 features hote hain,
# paper 12 bolta hai. Kaunsa feature drop kiya gaya, ye paper
# specify nahi karta. Hum saare 13 features use kar rahe hain -
# ye ek documented assumption hai (chhupaya nahi hai).

from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np


def load_imbalanced_heart():
    """
    UCI Statlog (Heart) dataset ko fetch karke, paper ke Table 1
    ke hisaab se binary imbalanced dataset banata hai.

    Returns:
        X (np.ndarray): shape (270, 13) - features
                        NOTE: paper 12 bolta hai, hum 13 use kar
                        rahe hain (dataset dropna ke baad bhi 13
                        hi hain - ye ek open discrepancy hai)
        y (np.ndarray): shape (270,) - labels
                        (1=minority/disease-presence, 0=majority)
    """
    # UCI repo ID 145 = Statlog (Heart)
    heart = fetch_ucirepo(id=145)

    X_df = heart.data.features
    y_df = heart.data.targets

    # Missing values wali rows hata do (agar hon)
    combined = pd.concat([X_df, y_df], axis=1).dropna()
    X_df = combined[X_df.columns]
    y_df = combined[y_df.columns]

    X = X_df.values.astype(np.float64)

    # UCI target: 1 = absence (majority), 2 = presence (minority)
    # Paper Table 1: minority class = "2", majority class = "1"
    raw_target = y_df.values.ravel()
    y = (raw_target == 2).astype(int)  # 1=minority(disease), 0=majority

    return X, y


if __name__ == "__main__":
    X, y = load_imbalanced_heart()
    n_minority = (y == 1).sum()
    n_majority = (y == 0).sum()
    IR = n_majority / n_minority

    print("Total samples:", len(y))
    print("Feature count:", X.shape[1], "(paper says 12 - see deviation note)")
    print("Minority (disease presence) samples:", n_minority)
    print("Majority (disease absence) samples:", n_majority)
    print(f"Imbalance Ratio (IR): {IR:.2f}   (paper's Table 1 value = 1.25)")