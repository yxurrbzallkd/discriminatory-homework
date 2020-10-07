import pandas as pd
from bayesian_classifier import BayesianClassifier
from main import process_data


classifier = BayesianClassifier()

print('training classifier')
X, y = process_data('train.csv')
classifier.fit(X, y)
print('accuracy on training data', classifier.score(X, y))
print(classifier.detailed_score(X, y))

print('\ntesting classifier')
X, y = process_data('test.csv')
print('accuracy on test data', classifier.score(X, y))
print(classifier.detailed_score(X, y))