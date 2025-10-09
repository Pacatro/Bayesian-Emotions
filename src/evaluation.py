from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

import config
from model import BayesianClsf


def kfold_cross_validation(
    X: list[str],
    y: list[str],
    k: int = 5,
    smoothing_factor: float = 1.0,
    min_len: int = 1,
    min_occurences: int = 1,
    verbose: bool = True,
) -> float:
    """Performs k-fold cross-validation for the BayesianClsf model.

    Args:
        X (list[str]): The list of reviews.
        y (list[str]): The list of sentiment labels.
        k (int): The number of folds for cross-validation. Defaults to 5.
        smoothing_factor (float): The smoothing factor for the model. Defaults to 1.0.
        min_len (int): The minimum word length for the model. Defaults to 1.
        min_occurences (int): The minimum word occurrences for the model. Defaults to 1.
        verbose (bool): Whether to print progress information. Defaults to True.

    Returns:
        float: The mean accuracy across all folds.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    accs = []

    if verbose:
        print(
            f"Running {k}-fold CV: min_len={min_len}, min_occurences={min_occurences}"
        )

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        if verbose:
            print(f"Training fold {fold}/{k}")

        X_train = [X[i] for i in train_idx]
        y_train = [y[i] for i in train_idx]
        X_val = [X[i] for i in val_idx]
        y_val = [y[i] for i in val_idx]

        model = BayesianClsf(
            smoothing_factor=smoothing_factor,
            min_len=min_len,
            min_occurences=min_occurences,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        accs.append(accuracy_score(y_val, preds))

    return np.mean(accs).item()


def compare_words_length(
    X: list[str],
    y: list[str],
    k: int = 5,
    lengths: list[int] | None = None,
) -> dict[int, float]:
    """Compares model performance for different minimum word lengths.

    This function runs k-fold cross-validation for a range of `min_len` values
    to find the optimal minimum word length for the model.

    Args:
        X (list[str]): The list of reviews.
        y (list[str]): The list of sentiment labels.
        k (int): The number of folds for cross-validation. Defaults to 5.
        lengths (list[int] | None): A list of minimum word lengths to test.
            If None, defaults to `list(range(1, 11))`.

    Returns:
        dict[int, float]: A dictionary mapping each minimum word length to its
            cross-validation accuracy.
    """
    if lengths is None:
        lengths = list(range(1, 11))

    if config.state["verbose"]:
        print(f"Comparing word lengths: {lengths}")

    return {
        L: kfold_cross_validation(X, y, k=k, min_len=L, min_occurences=1)
        for L in lengths
    }


def calc_metrics(
    preds: list[str],
    y_test: list[str],
) -> tuple[np.ndarray, float, float, float, float, float]:
    """Calculates and returns various performance metrics for the model.

    Args:
        model (BayesianClsf): The trained classifier model.
        preds (list[str]): The list of predicted sentiment labels.
        y_test (list[str]): The list of true test labels.

    Returns:
        tuple[np.ndarray, float, float, float, float, float]: A tuple containing
            the confusion matrix, accuracy, true positive rate (TPR),
            true negative rate (TNR), false positive rate (FPR), and
            false negative rate (FNR).
    """
    # Use labels ordered as [negative, positive] for TN, FP, FN, TP
    cm = confusion_matrix(y_test, preds, labels=["negative", "positive"])
    tn, fp, fn, tp = cm.ravel()
    total = len(y_test)
    acc = (tp + tn) / total
    tpr = tp / (tp + fn) if tp + fn else 0
    tnr = tn / (tn + fp) if tn + fp else 0
    fpr = fp / (fp + tn) if fp + tn else 0
    fnr = fn / (fn + tp) if fn + tp else 0
    return cm, acc, tpr, tnr, fpr, fnr
