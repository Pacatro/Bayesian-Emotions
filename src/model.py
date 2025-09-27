import re
from collections import Counter
import numpy as np


class BayesianClsf:
    """A Naive Bayes classifier for text classification.

    This class implements a simple Naive Bayes classifier from scratch,
    tailored for sentiment analysis of text data. It supports Laplace
    smoothing and feature selection based on minimum word length and
    occurrence frequency.
    """

    def __init__(
        self,
        smoothing_factor: float = 1.0,
        min_len: int = 1,
        min_occurences: int = 1,
    ):
        """Initializes the BayesianClsf classifier.

        Args:
            smoothing_factor (float): The smoothing factor for Laplace smoothing.
                Defaults to 1.0.
            min_len (int): The minimum length of a word to be considered a feature.
                Defaults to 1.
            min_occurences (int): The minimum number of times a word must appear
                in the training data to be considered a feature. Defaults to 1.
        """
        self.smoothing_factor = smoothing_factor
        self.min_len = min_len
        self.min_occurences = min_occurences
        self.relevant_features: list[str] = []
        self.positive_word_frequencies: dict[str, int] = {}
        self.negative_word_frequencies: dict[str, int] = {}
        self.positive_likelihoods: dict[str, float] = {}
        self.negative_likelihoods: dict[str, float] = {}
        self.prior_positive: float = 0.0
        self.prior_negative: float = 0.0

    def __clean_review(self, review: str) -> list[str]:
        """Cleans and tokenizes a review text.

        This method removes punctuation, converts the text to lowercase, splits it
        into words, and filters out words shorter than `self.min_len`.

        Args:
            review (str): The review text to clean.

        Returns:
            list[str]: A list of cleaned and filtered words.
        """
        cleaned = re.sub(r"[^a-zA-Z0-9\s]", "", review).lower().split()
        return [w for w in cleaned if len(w) >= self.min_len]

    def __extract_relevant_features(self, X_train: list[str]) -> None:
        """Extracts relevant features (words) from the training data.

        This method builds a vocabulary of relevant features by counting all words
        in the training data and filtering them based on `self.min_len` and
        `self.min_occurences`.

        Args:
            X_train (list[str]): The list of training reviews.
        """
        # Count all words
        all_words = [w for review in X_train for w in self.__clean_review(review)]
        word_count = Counter(all_words)
        # Keep only words meeting length and occurrence thresholds
        self.relevant_features = [
            word for word, count in word_count.items() if count >= self.min_occurences
        ]

    def __count_feature_frequencies(self, reviews: list[str]) -> dict[str, int]:
        """Counts the frequency of relevant features in a list of reviews.

        Args:
            reviews (list[str]): A list of reviews.

        Returns:
            dict[str, int]: A dictionary mapping each relevant feature to its
                frequency in the given reviews.
        """
        freqs = {word: 0 for word in self.relevant_features}
        for review in reviews:
            for word in self.__clean_review(review):
                if word in freqs:
                    freqs[word] += 1
        return freqs

    def fit(
        self,
        X_train: list[str],
        y_train: list[str],
    ) -> None:
        """Trains the Naive Bayes classifier.

        This method calculates the prior probabilities and likelihoods for each
        class (positive and negative) based on the training data.

        Args:
            X_train (list[str]): The list of training reviews.
            y_train (list[str]): The list of corresponding sentiment labels.
        """
        # Separate by class
        positive_reviews = [r for r, y in zip(X_train, y_train) if y == "positive"]
        negative_reviews = [r for r, y in zip(X_train, y_train) if y == "negative"]

        # Build vocabulary
        self.__extract_relevant_features(X_train)
        # Count occurrences in each class
        self.positive_word_frequencies = self.__count_feature_frequencies(
            positive_reviews
        )
        self.negative_word_frequencies = self.__count_feature_frequencies(
            negative_reviews
        )

        # Priors
        n_pos = len(positive_reviews)
        n_neg = len(negative_reviews)
        n_total = n_pos + n_neg
        self.prior_positive = n_pos / n_total
        self.prior_negative = n_neg / n_total

        # Likelihoods with Laplace smoothing
        V = len(self.relevant_features)
        for word in self.relevant_features:
            count_pos = self.positive_word_frequencies.get(word, 0)
            count_neg = self.negative_word_frequencies.get(word, 0)
            self.positive_likelihoods[word] = (count_pos + self.smoothing_factor) / (
                n_pos + self.smoothing_factor * V
            )
            self.negative_likelihoods[word] = (count_neg + self.smoothing_factor) / (
                n_neg + self.smoothing_factor * V
            )

    def predict(self, review: str) -> str:
        """Predicts the sentiment of a single review.

        This method calculates the posterior probability for each class and
        returns the class with the higher probability.

        Args:
            review (str): The review text to classify.

        Returns:
            str: The predicted sentiment ("positive" or "negative").
        """
        words = self.__clean_review(review)
        log_pos = np.log(self.prior_positive)
        log_neg = np.log(self.prior_negative)
        for w in words:
            if w in self.positive_likelihoods:
                log_pos += np.log(self.positive_likelihoods[w])
            if w in self.negative_likelihoods:
                log_neg += np.log(self.negative_likelihoods[w])
        return "positive" if log_pos > log_neg else "negative"
