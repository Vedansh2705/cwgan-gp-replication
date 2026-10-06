# data/scaling.py
# =============================================================
# Step 6b: Data Scaling (Phase 1 ka gap #4 - paper preprocessing
# specify nahi karta, ye hamara reasonable assumption hai)
# =============================================================
# Generator random noise (~-1 se +1 range) se shuru hota hai.
# Agar real data bade numbers (jaise 1-7) mein hai, Generator ko
# seekhna mushkil hota hai. Isliye hum data ko "scale" karte hain
# taaki sab features ka mean=0, std=1 ho jaaye - GAN ke liye
# seekhna aasan ho jaata hai.
#
# IMPORTANT: Scaler sirf TRAINING data par fit hota hai
# (data leakage na ho - jaisa humne train/test split mein discuss kiya)

from sklearn.preprocessing import StandardScaler


def fit_scaler(X_train):
    """
    Scaler ko training data par 'fit' karta hai - matlab
    seekhta hai ki har feature ka mean aur std kya hai.
    """
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def scale_data(scaler, X):
    """Data ko scaled version mein convert karta hai (mean=0, std=1)."""
    return scaler.transform(X)


def inverse_scale_data(scaler, X_scaled):
    """Scaled data ko wapas original range mein convert karta hai."""
    return scaler.inverse_transform(X_scaled)