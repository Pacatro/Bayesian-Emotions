from typing import Annotated
import typer
import time

from model import BayesianClsf
from data import load_data
from evaluation import compare_words_length, calc_metrics
import config

app = typer.Typer()


@app.command(
    help="Tests the model with new reviews, either predefined or from user input."
)
def test(
    interactive: Annotated[
        bool,
        typer.Option(
            "--interactive",
            "-i",
            help="Test model in interactive mode (Using user input)",
        ),
    ] = config.INTERACTIVE,
):
    X_train, y_train, _, _ = load_data(
        "data/movie_reviews.xlsx",
        verbose=config.state["verbose"],
        show_data_stats=config.state["show_data_stats"],
    )

    if config.state["verbose"]:
        print("Training model...\n")

    model = BayesianClsf()
    model.fit(X_train, y_train)

    if interactive:
        user_review = input("Write a new movie review: ")
        for pred in model.predict(user_review):
            print(f"Prediction: {pred}")
        return

    samples = [
        # negative sample
        "The Great Space Heist is a total disaster. The plot is confusing and filled with absurd twists... Avoid at all costs.",
        # positive sample
        "The Great Space Heist is a fantastic adventure! The plot is clever and full of unexpected twists... A must-watch.",
    ]

    preds = model.predict(samples)

    for rev, pred in zip(samples, preds):
        print("Review:", rev)
        print("Prediction:", pred, "\n")


@app.command(help="Performs a final evaluation using the optimal minimum word length.")
def eval(
    k: Annotated[int, typer.Option("--k", "-k", help="Number of folds")] = config.K,
    min_ocurrences: Annotated[
        int, typer.Option("--min-ocurrences", "-m", help="Minimum word occurences")
    ] = config.MIN_OCCURRENCES,
    smoothing_factor: Annotated[
        float, typer.Option("--smoothing-factor", "-s", help="Smoothing factor")
    ] = config.SMOOTHING_FACTOR,
    lengths: Annotated[
        list[int],
        typer.Option("--lengths", "-l", help="List of minimum word lengths to test"),
    ] = config.LENGTHS,
):
    X_train, y_train, X_test, y_test = load_data(
        "data/movie_reviews.xlsx",
        verbose=config.state["verbose"],
        show_data_stats=config.state["show_data_stats"],
    )

    start_time = time.time()

    # Find best min_len
    if config.state["verbose"]:
        print("Finding optimal min_len using cross-validation...")

    only_one_length = len(lengths) == 1

    results = (
        compare_words_length(X_train, y_train, k=k, lengths=lengths)
        if not only_one_length
        else {lengths[0]: 0.0}
    )

    best_len = max(results, key=results.get)

    if only_one_length and config.state["verbose"]:
        print("Only one length provided, skipping cross-validation.")

    # Train final model with optimal length
    if config.state["verbose"]:
        print(f"Training final model with optimal min_len: {best_len}")

    model = BayesianClsf(
        smoothing_factor=smoothing_factor,
        min_len=best_len,
        min_occurences=min_ocurrences,
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    cm, acc, tpr, tnr, fpr, fnr = calc_metrics(preds, y_test)

    eval_time = time.time() - start_time

    print("\nFinal Evaluation:")
    if only_one_length:
        print(f" Optimal min_len: {best_len} -> CV accuracy: {results[best_len]:.2%}\n")

    print(" Confusion Matrix:\n", cm)
    print(f" Accuracy: {acc:.2%}")
    print(f" TPR (Recall+): {tpr:.2%}")
    print(f" TNR (Recall-): {tnr:.2%}")
    print(f" FPR: {fpr:.2%}")
    print(f" FNR: {fnr:.2%}\n")
    print(f" Total evaluation time: {eval_time:.2f} seconds")
