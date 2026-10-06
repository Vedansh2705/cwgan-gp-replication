# data/load_iris.py
# =============================================================
# Step 1 + 2: Dataset Load karna aur Imbalance Create karna
# =============================================================
# Paper ke Table 1 ke hisaab se (ID 4: Iris):
#   Minority class = versicolor        (50 samples)
#   Majority class = all other (setosa + virginica) (100 samples)
#   Expected IR (imbalance ratio) = 2.00

from sklearn.datasets import load_iris
import pandas as pd
import numpy as np


def load_imbalanced_iris():
    """
    Iris dataset ko load karke, paper ke Table 1 ke hisaab se
    binary imbalanced dataset banata hai.

    Returns:
        X (np.ndarray): shape (150, 4) - features
        y (np.ndarray): shape (150,)  - labels (1=minority/versicolor, 0=majority)
    """
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target  # 0=setosa, 1=versicolor, 2=virginica

    # target==1 (versicolor) -> minority class -> label 1
    # baaki sab (setosa, virginica) -> majority class -> label 0
    df['label'] = (df['target'] == 1).astype(int)

    X = df[iris.feature_names].values
    y = df['label'].values
    return X, y


if __name__ == "__main__":
    X, y = load_imbalanced_iris()
    n_minority = (y == 1).sum()
    n_majority = (y == 0).sum()
    IR = n_majority / n_minority

    print("Total samples:", len(y))
    print("Minority (versicolor) samples:", n_minority)
    print("Majority (setosa+virginica) samples:", n_majority)
    print(f"Imbalance Ratio (IR): {IR:.2f}   (paper's Table 1 value = 2.00)")
