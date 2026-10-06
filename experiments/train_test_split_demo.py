import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.load_iris import load_imbalanced_iris
from sklearn.model_selection import train_test_split
import numpy as np


def get_train_test_splits(X, y, n_repeats=5, test_size=0.2):
    splits = []
    for repeat_idx in range(n_repeats):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=repeat_idx,
            stratify=y
        )
        splits.append((X_train, X_test, y_train, y_test))
    return splits


if __name__ == "__main__":
    X, y = load_imbalanced_iris()
    splits = get_train_test_splits(X, y, n_repeats=5, test_size=0.2)

    print(f"Total {len(splits)} splits bane (paper: 5 repeats)\n")

    for i, (X_train, X_test, y_train, y_test) in enumerate(splits):
        train_minority = (y_train == 1).sum()
        train_majority = (y_train == 0).sum()
        test_minority = (y_test == 1).sum()
        test_majority = (y_test == 0).sum()

        print(f"Repeat {i+1}:")
        print(f"  Train: {len(X_train)} samples "
              f"(minority={train_minority}, majority={train_majority})")
        print(f"  Test:  {len(X_test)} samples "
              f"(minority={test_minority}, majority={test_majority})")