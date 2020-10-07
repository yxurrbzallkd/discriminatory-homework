import os
import re

from bayesian_classifier import BayesianClassifier
import pandas as pd


def delete_symbols(msg: str):
    """
    Substitutes symbols with spaces.
    :param msg: str
    :return: str
    """

    symbols = re.compile(r"[-!$%^&*()_+©|~#=.`{}\[\]:'\";<>?,ʼ/]|[^\w]")

    return re.sub(symbols, " ", msg)


def remove_stop_words(data, stop_words):
    """

    :return: cleaned text from stop_words
    """
    words = data.split()

    new_text = ""
    for w in words:
        w = w.strip()

        if w not in stop_words and len(w) > 1:
            new_text = new_text + " " + w

    return new_text


def split_tweet(tweet):
    """split twin into words"""
    if pd.isnull(tweet):
        return ""

    return tweet.split()


def process_data(data_file):
    """
    Function for data processing and split it into X and y sets.
    :param data_file: str - path to dataframe
    """
    df = pd.read_csv(data_file)
    with open(os.path.join("static", "stop_words.txt"), "r", encoding="utf-8") as f:
        stop_words = []
        for word in f.readlines():
            stop_words.append(word[:-1])

    df["preprocessed_tweet"] = ''
    for index, row in df.iterrows():
        row.preprocessed_tweet = delete_symbols(row.tweet)
        row.preprocessed_tweet = remove_stop_words(row.preprocessed_tweet, stop_words)
        df.at[index, "preprocessed_tweet"] = row.preprocessed_tweet

    df.to_csv(data_file[:-4] + "_preprocessed.csv", index=False)


def get_processed_df(data_file):
    """
    :return: pd.DataFrame|list, pd.DataFrame|list - X and y data frames or lists
    """
    df = pd.read_csv(data_file[:-4] + "_preprocessed.csv")
    return [split_tweet(i) for i in df['preprocessed_tweet']], df['label']


if __name__ == "__main__":
    process_data(os.path.join("1-discrimination", "train.csv"))
    process_data(os.path.join("1-discrimination", "test.csv"))

    train_X, train_y = get_processed_df(os.path.join("1-discrimination", "train.csv"))
    test_X, test_y = get_processed_df(os.path.join("1-discrimination", "test.csv"))

    classifier = BayesianClassifier()
    classifier.fit(train_X, train_y)
    classifier.predict_prob(test_X[0], test_y[0])

    print("model score: ", classifier.score(test_X, test_y))
