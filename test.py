import pandas as pd
from bayesian_classifier import BayesianClassifier

classifier = BayesianClassifier()

print('training classifier')
train_df = pd.read_csv('train.csv')
classifier.fit(train_df['tweet'], train_df['label'])
print('accuracy on training data', classifier.score(train_df['tweet'], train_df['label']))
print(classifier.detailed_score(train_df['tweet'], train_df['label']))

print('\ntesting classifier')
test_df = pd.read_csv('test.csv')
print('accuracy on test data', classifier.score(test_df['tweet'], test_df['label']))
print(classifier.detailed_score(test_df['tweet'], test_df['label']))