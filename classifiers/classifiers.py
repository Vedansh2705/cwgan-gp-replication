# classifiers/classifiers.py
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


def get_classifiers(random_state=42):
    classifiers = {
        "KNN": KNeighborsClassifier(),
        "LR": LogisticRegression(random_state=random_state, max_iter=1000),
        "RF": RandomForestClassifier(random_state=random_state),
        "DT": DecisionTreeClassifier(random_state=random_state),
        "SVM": SVC(kernel="rbf", probability=True, random_state=random_state)
    }
    return classifiers


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

    from data.load_iris import load_imbalanced_iris
    from experiments.train_test_split_demo import get_train_test_splits
    from baselines.smote import apply_smote

    X, y = load_imbalanced_iris()
    splits = get_train_test_splits(X, y, n_repeats=5, test_size=0.2)
    X_train, X_test, y_train, y_test = splits[0]

    X_resampled, y_resampled = apply_smote(X_train, y_train, k_neighbors=5, random_state=0)

    classifiers = get_classifiers()

    print("Classifier performance (SMOTE-oversampled training data par):\n")
    for name, clf in classifiers.items():
        clf.fit(X_resampled, y_resampled)
        accuracy = clf.score(X_test, y_test)
        print(f"  {name:5s}: test accuracy = {accuracy:.3f}")