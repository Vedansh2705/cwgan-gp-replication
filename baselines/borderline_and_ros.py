# baselines/borderline_and_ros.py
# =============================================================
# Baselines: Borderline-SMOTE aur ROS (Paper Table 3)
# =============================================================
# Borderline-SMOTE: sirf un minority samples ko oversample karta
#   hai jo "borderline" (decision boundary ke paas) hain, kyunki
#   wahi galat classify hone ka sabse zyada risk hota hai.
#   Paper settings: k_neighbors=5, m_neighbors=10
#
# ROS (Random OverSampling): sabse simple - bas minority class
#   ke existing samples ko RANDOMLY DUPLICATE karta hai (koi
#   naya point nahi banata).

from imblearn.over_sampling import BorderlineSMOTE, RandomOverSampler


def apply_borderline_smote(X_train, y_train, k_neighbors=5, m_neighbors=10, random_state=None):
    """
    Paper ke Table 3 settings: k_neighbors=5, m_neighbors=10
    """
    bsmote = BorderlineSMOTE(
        k_neighbors=k_neighbors,
        m_neighbors=m_neighbors,
        random_state=random_state
    )
    X_resampled, y_resampled = bsmote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled


def apply_ros(X_train, y_train, random_state=None):
    """
    Random Oversampling - existing minority samples ko duplicate
    karta hai jab tak balance na ho jaaye.
    """
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

    print("=" * 50)
    print("Borderline-SMOTE")
    print("=" * 50)
    print(f"PEHLE: Minority={ (y_train==1).sum() }, Majority={ (y_train==0).sum() }")
    X_bs, y_bs = apply_borderline_smote(X_train, y_train, random_state=0)
    print(f"BAAD:  Minority={ (y_bs==1).sum() }, Majority={ (y_bs==0).sum() }, Total={len(y_bs)}")

    print()
    print("=" * 50)
    print("ROS (Random Oversampling)")
    print("=" * 50)
    print(f"PEHLE: Minority={ (y_train==1).sum() }, Majority={ (y_train==0).sum() }")
    X_ros, y_ros = apply_ros(X_train, y_train, random_state=0)
    print(f"BAAD:  Minority={ (y_ros==1).sum() }, Majority={ (y_ros==0).sum() }, Total={len(y_ros)}")