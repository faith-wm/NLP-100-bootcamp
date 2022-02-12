
import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.stem import PorterStemmer

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn import  preprocessing
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2

import random
random.seed(10)

newsCorpora='NewsAggregatorDataset/newsCorpora.csv'
pageSessions= 'NewsAggregatorDataset/2pageSessions.csv'

#50. Download and Preprocess DatasetPermalink
read_newsCorpora=pd.read_csv(newsCorpora, delimiter='\t',quoting=3,
                             names=['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP'])

publishers = ['Reuters','Huffington Post','Businessweek','Contactmusic.com','Daily Mail']

filtered_Df=read_newsCorpora[read_newsCorpora['PUBLISHER'].isin(publishers)]
filtered_Df.sample(frac=10,replace=True).reset_index()

train_len=int(len(filtered_Df)*0.8)
valid_len=int(len(filtered_Df)*0.9)

train_data= filtered_Df[:train_len]
valid_data=filtered_Df[train_len:valid_len]
test_data=filtered_Df[valid_len:]

print(len(filtered_Df),len(train_data),len(valid_data),len(test_data))


train_data = train_data[['CATEGORY','TITLE']]
train_data.to_csv('NewsAggregatorDataset/train.txt',index=None,header=None,sep='\t')

valid_data = valid_data[['CATEGORY','TITLE']]
valid_data.to_csv('NewsAggregatorDataset/valid.txt',index=None,header=None,sep='\t')

test_data = test_data[['CATEGORY','TITLE']]
test_data.to_csv('NewsAggregatorDataset/test.txt',index=None,header=None,sep='\t')


# print('\ntrain instances\n',train_data['CATEGORY'].value_counts())
# print('\nvalid instances\n',valid_data['CATEGORY'].value_counts())
# print('\ntest instances\n',test_data['CATEGORY'].value_counts())


# 51. Feature extraction
# https://www.analyticsvidhya.com/blog/2018/04/a-comprehensive-guide-to-understand-and-implement-text-classification-in-python/
def clean_text(text):
    text = text.lower()

    #remove punctuations
    for ch in string.punctuation:
        text = text.replace(ch,'')
    text = re.sub("[0-9]+", "||DIG||", text)
    text = re.sub(' +', ' ', text)

    #remove stopwords
    # ps = PorterStemmer()
    stop_words = set(stopwords.words("english"))

    text_tokens = word_tokenize(text)
    # stem_tokens = [ps.stem(w) for w in text_tokens]
    remove_stop = [i for i in text_tokens if i not in stop_words]

    return ' '.join(remove_stop)


train_data['clean'] = [clean_text(txt) for txt in train_data['TITLE'] ]
valid_data['clean']= [clean_text(txt) for txt in valid_data['TITLE'] ]
test_data['clean']= [clean_text(txt) for txt in test_data['TITLE'] ]

filtered_Df['clean']= [clean_text(txt) for txt in filtered_Df['TITLE'] ]

X_train,y_train= train_data['clean'], train_data['CATEGORY']
X_valid,y_valid = valid_data['clean'], valid_data['CATEGORY']
X_test,y_test = test_data['clean'], test_data['CATEGORY']



#F1. count vectors  --> matrix with frequency of word in dataset
count_vector = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vector.fit(filtered_Df['clean'])

X_train_count = count_vector.transform(X_train)
X_valid_count = count_vector.transform(X_valid)
X_test_count = count_vector.transform(X_valid)

#F2 tf-idf vectors - importance of term in a document
#word level tfidf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(filtered_Df['clean'])

X_train_tfidf = tfidf_vect.transform(X_train)
X_valid_tfidf = tfidf_vect.transform(X_valid)
X_test_tfidf = tfidf_vect.transform(X_test)

#ngram level tfidf
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram.fit(filtered_Df['clean'])

X_train_tfidf_ngram = tfidf_vect_ngram.transform(X_train)
X_valid_tfidf_ngram = tfidf_vect_ngram.transform(X_valid)
X_test_tfidf_ngram = tfidf_vect_ngram.transform(X_test)


#character level tfidf
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram_chars.fit(filtered_Df['clean'])

X_train_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(X_train)
X_valid_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(X_valid)
X_test_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(X_test)


#encoding labels
# encoder = preprocessing.LabelEncoder()
# y_train = encoder.fit_transform(y_train)
# y_valid = encoder.fit_transform(y_valid)
# y_test = encoder.fit_transform(y_test)



# 52. Training and  53. prediction
model = LogisticRegression(solver='liblinear')
model.fit(X_train_tfidf,y_train)

valid_predictions = model.predict(X_valid_tfidf)
test_predictions = model.predict(X_test_tfidf)


#54. Accuracy
test_accuracy= metrics.accuracy_score(test_predictions, y_test)
print ("test accuracy LR, Count Vectors accuracy: ", test_accuracy)

valid_accuracy= metrics.accuracy_score(valid_predictions, y_valid)
print ("valid accuracy LR, Count Vectors accuracy: ", valid_accuracy)

#55. confusion matrix

print("test confusion matrix:\n ",
      metrics.confusion_matrix(y_test, test_predictions,))

print("valid confusion matrix:\n ",
      metrics.confusion_matrix(y_valid, valid_predictions))

#56. P,R,F1
print("test, P,R,F1:\n ",
      metrics.classification_report(y_test, test_predictions))

print("valid, P,R,F1:\n ",
      metrics.classification_report(y_test, valid_predictions))

#57. Feature weights
print('features \n')
feature_importance = model.coef_[0]   #(n_classes,n_features)
feature_importance=sorted(feature_importance, reverse=True)
print(feature_importance[:10])





#58. Regularization parameters
model2 = LogisticRegression(solver='sag',penalty='l2',multi_class='ovr',max_iter=1000,warm_start=True)
model2.fit(X_train_tfidf,y_train)

valid_predictions = model2.predict(X_valid_tfidf)
test_predictions = model2.predict(X_test_tfidf)

test_accuracy= metrics.accuracy_score(test_predictions, y_test)
print ("test accuracy LR, Count Vectors accuracy: ", test_accuracy)

valid_accuracy= metrics.accuracy_score(valid_predictions, y_valid)
print ("valid accuracy LR, Count Vectors accuracy: ", valid_accuracy)


print('===================================\n')
#59. Hyperparameter tuning
from sklearn.naive_bayes import GaussianNB
from sklearn import svm

model2 = svm.SVC()
model2.fit(X_train_tfidf,y_train)

valid_predictions = model2.predict(X_valid_tfidf)
test_predictions = model2.predict(X_test_tfidf)

test_accuracy= metrics.accuracy_score(test_predictions, y_test)
print ("test accuracy LR, Count Vectors accuracy: ", test_accuracy)

valid_accuracy= metrics.accuracy_score(valid_predictions, y_valid)
print ("valid accuracy LR, Count Vectors accuracy: ", valid_accuracy)


#GridSearchCV









