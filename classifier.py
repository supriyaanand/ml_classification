import pandas as pd
import math
import string
from collections import Counter
import requests
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import Imputer
import numpy

products = pd.read_csv('./amazon_baby.csv')

def remove_punctuation(text):
	if isinstance(text, str):
		return text.translate(None, string.punctuation) 
	else:
		return text



products['review_clean'] = products['review'].apply(remove_punctuation)

products = products.fillna({'review_clean':'NA'})  # fill in N/A's in the review column

products = products[products.rating != 3] # Ignore neutral ratings

products['sentiment'] = products['rating'].apply(lambda rating : +1 if rating > 3 else -1) # Assign sentiment tags

train_data = []
link = "https://s3.amazonaws.com/static.dato.com/files/coursera/course-3/indices-json/module-2-assignment-train-idx.json"
f = requests.get(link)
index_list = json.loads(f.text)
for i in index_list:
	train_data.append(products.iloc[i])

train_data = pd.DataFrame(train_data)

test_data = []
link = "https://s3.amazonaws.com/static.dato.com/files/coursera/course-3/indices-json/module-2-assignment-test-idx.json"
f = requests.get(link)
index_list = json.loads(f.text)
for i in index_list:
	test_data.append(products.iloc[i])

test_data = pd.DataFrame(test_data)

# Bag of words training model
vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
train_matrix = vectorizer.fit_transform(train_data['review_clean'])
test_matrix = vectorizer.transform(test_data['review_clean'])


sentiment_model = LogisticRegression()
sentiment_model.fit(train_matrix, train_data['sentiment'])

dx = sentiment_model.__dict__
coefs = dx['coef_'][0]

print "Number of co-efficients ", len(coefs)

count = 0
for co in coefs:
	if co >= 0:
		count += 1

print "Number of non negative coeffs ", count

sample_test_data = test_data[10:13]
print sample_test_data

def probability(score):
	return (1 / (1 + numpy.exp(-score)))

sample_test_matrix = vectorizer.transform(sample_test_data['review_clean'])
scores = sentiment_model.decision_function(sample_test_matrix)
print scores
print sentiment_model.predict(sample_test_matrix)

predictions = map(sentiment_model.predict())



