import pandas as pd


def load_data(
    path: str, verbose: bool = False, show_data_stats: bool = False
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Loads and splits the movie review data from an Excel file.

    Args:
        path (str): The path to the Excel file.
        verbose (bool): Whether to print data loading and split information.
            Defaults to True.
        show_data_stats (bool): Whether to print data statistics. Defaults to True.

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
    if show_data_stats:
        for split in ["train", "test"]:
            print(f"Positive {split} reviews: {counts.loc[split].get('positive', 0)}")
            print(f"Negative {split} reviews: {counts.loc[split].get('negative', 0)}")
        print()

    return X_train, y_train, X_test, y_test
