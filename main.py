from bayesian_classifier import BayesianClassifier
import pandas as pd


def split_tweet(tweet):
    '''split twin into words'''
    return tweet.split()

def process_data(data_file):
    """
    Function for data processing and split it into X and y sets.
    :param data_file: str - train data
    :return: pd.DataFrame|list, pd.DataFrame|list - X and y data frames or lists
    """
    df = pd.read_csv(data_file)
    return [split_tweet(i) for i in df['tweet']], df['label']


if __name__ == "__main__":
    train_X, train_y = process_data("train.csv")
    test_X, test_y = process_data("test.csv")

    classifier = BayesianClassifier()
    classifier.fit(train_X, train_y)
    classifier.predict_prob(test_X[0], test_y[0])

    print("model score: ", classifier.score(test_X, test_y))
