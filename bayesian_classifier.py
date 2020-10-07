import pandas as pd

class BayesianClassifier:
    """
    Implementation of Naive Bayes classification algorithm.
    """
    def __init__(self):
        pass

    def split_tweet(self, tweet: str) -> list:
        '''preprocess tweet - extract words'''
        return tweet.split()

    def count_df(self, tweets, labels) -> (dict, int, int):
        '''
        returns:
            words_dict = {word (str): {'discrim': int, 'neutral': int}}
            discrim, neutral: (int, int) - total number of discriminatory\neutral tweets
        '''
        words_dict = {}
        discrim = neutral = 0

        for tweet, label in zip(tweets, labels):
            for word in self.split_tweet(tweet):
                if word not in words_dict:
                    words_dict[word] = {'neutral': 0, 'discrim': 0}

                words_dict[word]['neutral'] += int(label=='neutral')
                words_dict[word]['discrim'] += int(label=='discrim')

                discrim += int(label=='discrim')
                neutral += int(label=='neutral')

        return words_dict, discrim, neutral

    def make_likelyhood_table(self, dictionary: dict, discrim: int, neutral: int):
        '''
        Transform:
            words_dict = {word (str): {'discrim': int, 'neutral': int}}
            discrim, neutral: (int, int) - total number of discriminatory\neutral tweet
        Into:
            table: {word (str): {'neutral': float, 'discrim': float}}
            word - a word from words_dict
            table[word]['neutral'\'discrim'] - neutral\discriminatory tweets
                                                with this word over total number
                                                of neutral\discriminatory tweets
        '''
        table = {word: {'neutral': None, 'discrim': None} for word in dictionary}
        for word in dictionary:
            n, d = dictionary[word]['neutral'], dictionary[word]['discrim']
            table[word]['neutral'] = (n+1)/(neutral+len(dictionary))
            table[word]['discrim'] = (d+1)/(discrim+len(dictionary))
        return table

    def calculate_probability(self, tweet: str) -> (float, float):
        '''
        returns:
            probabilities the tweet is discriminatory\neutral
        '''
        words = self.split_tweet(tweet)
        p_neutral = self.neutral_probability  #basic assumption
        p_discrim = self.discrim_probability  #basic assumption
        for word in set(words):
            if word in self.likelyhood_table:
                p_neutral *= self.likelyhood_table[word]['neutral']
                p_discrim *= self.likelyhood_table[word]['discrim']
            p_neutral, p_discrim = p_neutral/(p_neutral+p_discrim), p_discrim/(p_neutral+p_discrim)
        return p_discrim, p_neutral

    def fit(self, X: list, y: list):
        """
        Fit Naive Bayes parameters according to train data X and y.
        :param X: pd.DataFrame|list - train input/messages
        :param y: pd.DataFrame|list - train output/labels
        :return: None
        """
        dictionary, discrim, neutral = self.count_df(X, y)
        self.neutral_probability = neutral/(discrim+neutral)
        self.discrim_probability = discrim/(discrim+neutral)
        self.likelyhood_table = self.make_likelyhood_table(dictionary, discrim, neutral)

    def predict_prob(self, message: str, label: str) -> float:
        """
        Calculate the probability that a given label can be assigned to a given message.
        :param message: str - input message
        :param label: str - label
        :return: float - probability P(label|message)
        """
        p_discrim, p_neutral = self.calculate_probability(message)
        return p_discrim if label=='discrim' else p_neutral

    def predict(self, message: str) -> str:
        """
        Predict label for a given message.
        :param message: str - message
        :return: str - label that is most likely to be truly assigned to a given message
        """
        p_discrim, p_neutral = self.calculate_probability(message)
        return 'discrim' if p_discrim > p_neutral else 'neutral'

    def detailed_score(self, X: list, y: list):
        accuracy = {'true': {'positives': 0, 'negatives': 0}, 'false': {'positives': 0, 'negatives': 0}}
        total = 0
        for tweet, label in zip(X, y):
        	total += 1
        	estimated_label = self.predict(tweet)
        	if estimated_label == label:
        		if label=='discrim':
        			accuracy['true']['positives'] += 1
        		else:
        			accuracy['true']['negatives'] += 1
        	else:
        		if label=='discrim':
        			accuracy['false']['positives'] += 1
        		else:
        			accuracy['false']['negatives'] += 1

        return pd.DataFrame(accuracy)

    def score(self, X: list, y: list)-> float:
        """
        Return the mean accuracy on the given test data and labels - the efficiency of a trained model.
        :param X: pd.DataFrame|list - test data - messages
        :param y: pd.DataFrame|list - test labels
        :return:
        """
        correctly_calssified = 0
        for tweet, label in zip(X, y):
            predicted_label = self.predict(tweet)
            if label == predicted_label:
                correctly_calssified += 1
        return correctly_calssified/len(X)
