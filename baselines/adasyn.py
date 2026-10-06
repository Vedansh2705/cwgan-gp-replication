# baselines/adasyn.py
# =============================================================
# Baseline: ADASYN (Paper Table 3)
# =============================================================
# ADASYN, SMOTE se milta-julta hai, lekin har minority sample ko
# uski "difficulty" ke hisaab se weight deta hai.
#
# Paper ke exact settings (Table 3): n_neighbors = 5
#
# KNOWN EDGE CASE: Jab classes already lagbhag balanced hon (jaise
# Liver dataset, IR=1.23), ADASYN kabhi-kabhi decide karta hai ki
# koi synthetic sample generate karne ki zaroorat nahi, aur
# ValueError deta hai ("No samples will be generated"). Isse
# handle karne ke liye, hum is case mein ROS (Random Oversampling)
# par fallback karte hain - ye documented deviation hai.

from imblearn.over_sampling import ADASYN, RandomOverSampler


def apply_adasyn(X_train, y_train, n_neighbors=5, random_state=None):
    """
    Training set par ADASYN oversampling apply karta hai.
    Agar ADASYN fail ho jaaye (nearly-balanced data ki wajah se),
    ROS par fallback karta hai.
    """
    try:
        adasyn = ADASYN(n_neighbors=n_neighbors, random_state=random_state)
        X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)
        return X_resampled, y_resampled
    except ValueError as e:
        print(f"    [WARNING] ADASYN fail hua ({e}); ROS par fallback kar rahe hain")
        ros = RandomOverSampler(random_state=random_state)
        X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
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

    print("ADASYN se PEHLE:")
    print(f"  Minority: {(y_train==1).sum()}, Majority: {(y_train==0).sum()}")

    X_resampled, y_resampled = apply_adasyn(X_train, y_train, n_neighbors=5, random_state=0)

    print("\nADASYN ke BAAD:")
    print(f"  Minority: {(y_resampled==1).sum()}, Majority: {(y_resampled==0).sum()}")
    print(f"  Total samples: {len(y_resampled)} (pehle {len(y_train)} the)")