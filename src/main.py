import pandas as pd

from model import BayesianClsf
from evaluation import final_eval_with_optimal_word_length


def get_data(
    path: str, verbose: bool = True
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Loads and splits the movie review data from an Excel file.

    Args:
        path (str): The path to the Excel file.
        verbose (bool): Whether to print data loading and split information.
            Defaults to True.

    Returns:
        tuple[list[str], list[str], list[str], list[str]]: A tuple containing
            X_train, y_train, X_test, and y_test lists.
    """
    if verbose:
        print("Loading data...")

    data = pd.read_excel(path)
    X_train = data[data["Split"] == "train"]["Review"].tolist()
    y_train = data[data["Split"] == "train"]["Sentiment"].tolist()
    X_test = data[data["Split"] == "test"]["Review"].tolist()
    y_test = data[data["Split"] == "test"]["Sentiment"].tolist()

    # Count splits by sentiment
    counts = data.groupby(["Split", "Sentiment"]).size().unstack(fill_value=0)
    if verbose:
        for split in ["train", "test"]:
            print(f"Positive {split} reviews: {counts.loc[split].get('positive', 0)}")
            print(f"Negative {split} reviews: {counts.loc[split].get('negative', 0)}")
        print()

    return X_train, y_train, X_test, y_test


def test_with_new_reviews(model: BayesianClsf, interactive: bool = False):
    """Tests the model with new reviews, either predefined or from user input.

    Args:
        model (BayesianClsf): The trained classifier model.
        interactive (bool): If True, prompts the user for a review.
            Otherwise, uses predefined sample reviews. Defaults to False.
    """
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
        print("Review:\n", rev)
        print("Prediction:", model.predict(rev), "\n")


def main():
    X_train, y_train, X_test, y_test = get_data("data/movie_reviews.xlsx")

    try:
        while True:
            choice = int(
                input(
                    "1) Random samples\n2) Your review\n3) Optimal eval\n(0 to quit): "
                )
            )
            match choice:
                case 1:
                    print("\nTraining new model...\n")
                    base_model = BayesianClsf()
                    base_model.fit(X_train, y_train)
                    test_with_new_reviews(base_model)
                case 2:
                    print("\nTraining new model...\n")
                    base_model = BayesianClsf()
                    base_model.fit(X_train, y_train)
                    test_with_new_reviews(base_model, interactive=True)
                case 3:
                    final_eval_with_optimal_word_length(
                        X_train, y_train, X_test, y_test
                    )
                case 0:
                    break
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print("Error:", e)
    finally:
        print("Exiting...")


if __name__ == "__main__":
    main()
