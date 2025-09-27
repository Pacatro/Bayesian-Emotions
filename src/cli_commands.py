from typing import Annotated
import typer

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
        print(f"Prediction: {model.predict(user_review)}")
        return

    samples = [
        # negative sample
        "The Great Space Heist is a total disaster. The plot is confusing and filled with absurd twists... Avoid at all costs.",
        # positive sample
        "The Great Space Heist is a fantastic adventure! The plot is clever and full of unexpected twists... A must-watch.",
    ]

    for rev in samples:
        print("Review:", rev)
        print("Prediction:", model.predict(rev), "\n")


@app.command(help="Performs a final evaluation using the optimal minimum word length.")
def eval(
    k: Annotated[int, typer.Option("--k", "-k", help="Number of folds")] = config.K,
    min_ocurrences: Annotated[
        int, typer.Option("--min-ocurrences", "-m", help="Minimum word occurences")
    ] = config.MIN_OCCURRENCES,
    smoothing_factor: Annotated[
        float, typer.Option("--smoothing-factor", "-s", help="Smoothing factor")
    ] = config.SMOOTHING_FACTOR,
):
    X_train, y_train, X_test, y_test = load_data(
        "data/movie_reviews.xlsx",
        verbose=config.state["verbose"],
        show_data_stats=config.state["show_data_stats"],
    )

    # Find best min_len
    results = compare_words_length(X_train, y_train, k=k)
    best_len = max(results, key=results.get)

    # Train final model with optimal length
    if config.state["verbose"]:
        print(f"Training final model with optimal min_len: {best_len}")

    model = BayesianClsf(
        smoothing_factor=smoothing_factor,
        min_len=best_len,
        min_occurences=min_ocurrences,
    )
    model.fit(X_train, y_train)

    cm, acc, tpr, tnr, fpr, fnr = calc_metrics(model, X_test, y_test)

    print("\nFinal Evaluation:")
    print(f" Optimal min_len: {best_len} -> CV accuracy: {results[best_len]:.2%}")
    print(" Confusion Matrix:\n", cm)
    print(f" Accuracy: {acc:.2%}")
    print(f" TPR (Recall+): {tpr:.2%}")
    print(f" TNR (Recall-): {tnr:.2%}")
    print(f" FPR: {fpr:.2%}")
    print(f" FNR: {fnr:.2%}\n")
