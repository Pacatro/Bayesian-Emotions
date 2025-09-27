# Bayesian Emotions: Sentiment Analysis

This project is a simple implementation of a Naive Bayes classifier for sentiment analysis on movie reviews. The model is built from scratch using Python and standard libraries.

## Features

- **Naive Bayes Classifier**: A custom-built classifier for text.
- **Data Preprocessing**: Cleans and tokenizes text data.
- **Hyperparameter Tuning**: Includes functionality to find the optimal minimum word length for feature selection.
- **Model Evaluation**: Uses k-fold cross-validation and calculates various metrics like accuracy, TPR, TNR, etc.
- **Interactive Mode**: Allows users to input their own reviews for prediction.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/Pacatro/bayesian_emotions.git
    cd bayesian_emotions
    ```

2. This project uses `uv` for package management. You can run the following command to setup the entire project:

    ```bash
    uv sync
    ```

3. Run the project:

    ```bash
    uv run src/main.py
    ```

## Usage

You will be presented with a menu:

- **1) Random samples**: Trains a model and predicts the sentiment of a few predefined sample reviews.
- **2) Your review**: Trains a model and prompts you to enter your own review for sentiment prediction.
- **3) Optimal eval**: Performs a more thorough evaluation by first finding the optimal minimum word length via cross-validation on the training set, and then reports detailed metrics on the test set.
- **q) to quit**: Exits the program.

## Project Structure

- `data/movie_reviews.xlsx`: The dataset containing movie reviews, their sentiment, and a train/test split.
- `src/main.py`: The main entry point of the application.
- `src/model.py`: Contains the `BayesianClsf` class, the core Naive Bayes classifier.
- `src/evaluation.py`: Contains functions for model evaluation and hyperparameter tuning.
- `pyproject.toml`: Project metadata and dependencies.

## License

[MIT](https://opensource.org/license/mit/) - Created by [**Paco Algar**](https://github.com/Pacatro).

