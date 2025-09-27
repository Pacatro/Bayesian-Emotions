# Bayesian Emotions: Sentiment Analysis

Simple implementation of a Naive Bayes classifier for sentiment analysis on movie reviews built from scratch.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/Pacatro/Bayesian-Emotions.git
    cd Bayesian-Emotions
    ```

2. This project uses [`uv`](https://docs.astral.sh/uv/) for package management. You can run the following command to setup the entire project:

    ```bash
    uv sync
    ```

3. Run the project:

    ```bash
    uv run src/main.py --help
    ```

## Usage

The application provides a CLI with two main commands: `test` and `eval`.

```bash
Usage: main.py [OPTIONS] COMMAND [ARGS]...

Options:
  -v, --verbose          Verbose mode
  -s, --show-data-stats  Show data stats
  -h, --help             Show this message and exit.

Commands:
  test  Tests the model with new reviews, either predefined or from user...
  eval  Performs a final evaluation using the optimal minimum word length.
```

You can use global options like `--verbose` (`-v`) to see more detailed output and `--show-data-stats` (`-s`) to display dataset statistics upon loading.

### Test the model

The `test` command trains a model and predicts the sentiment of reviews.

```bash
Usage: main.py test [OPTIONS]

  Tests the model with new reviews, either predefined or from user input.

Options:
  -i, --interactive  Test model in interactive mode (Using user input)
  -h, --help         Show this message and exit.
```

- **Test with predefined sample reviews**:

    ```bash
    uv run src/main.py test
    ```

- **Test with your own review in interactive mode**:

    ```bash
    uv run src/main.py test --interactive
    ```

    You will be prompted to enter a review.

### Evaluate the model

The `eval` command performs a more thorough evaluation. It first finds the optimal minimum word length via cross-validation on the training set, and then reports detailed metrics on the test set.

```bash
Usage: main.py eval [OPTIONS]

  Performs a final evaluation using the optimal minimum word length.

Options:
  -k, --k INTEGER               Number of folds  [default: 5]
  -m, --min-ocurrences INTEGER  Minimum word occurences  [default: 1]
  -s, --smoothing-factor FLOAT  Smoothing factor  [default: 1.0]
  -h, --help                    Show this message and exit.
```

- **Run the evaluation with default parameters**:

    ```bash
    uv run src/main.py eval
    ```

- **Customize evaluation parameters**:
    You can customize the number of folds (`-k`), minimum word occurrences (`-m`), and the smoothing factor (`-s`).

    ```bash
    uv run src/main.py eval -k 10 -m 2 -s 1.5
    ```

## License

[MIT](https://opensource.org/license/mit/) - Created by [**Paco Algar**](https://github.com/Pacatro).

