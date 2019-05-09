
# coding: utf-8

# # James Schallert
# # Data Mining
# # HW 5

# In[1]:


import numpy as np
import pandas as pd
import json
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import string

from sklearn.linear_model import LogisticRegression
from sklearn import svm
import pylab as pl
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV

import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


tweets = []
test_tweets = []
pos_tweets = 0
neg_tweets = 0
table = str.maketrans({key: None for key in string.punctuation})

# Reading all of the tweets from my files created from crawling Twitter.
with open('m.txt') as f:
        for line in f:
            rawTweet = json.loads(line)
            rawTweet['text'] = rawTweet['text'].translate(table)
            if (rawTweet['relevant']==False):
                if (neg_tweets < 250):
                    tweet = [0,rawTweet['text'].lower().split()]
                    tweets.append(tweet)
                    neg_tweets+=1
                else:
                    tweet = [0,rawTweet['text'].lower().split()]
                    test_tweets.append(tweet)
                    neg_tweets+=1
            elif (rawTweet['relevant']==True):
                if (pos_tweets < 250):
                    tweet = [1,rawTweet['text'].lower().split()]
                    tweets.append(tweet)
                    pos_tweets+=1
                else:
                    tweet = [1,rawTweet['text'].lower().split()]
                    test_tweets.append(tweet)
                    pos_tweets+=1
            else:
                break
                
with open('d.txt') as f:
        for line in f:
            rawTweet = json.loads(line)
            rawTweet['text'] = rawTweet['text'].translate(table)
            if (rawTweet['relevant']==False):
                if (neg_tweets < 250):
                    tweet = [0,rawTweet['text'].lower().split()]
                    tweets.append(tweet)
                    neg_tweets+=1
                else:
                    tweet = [0,rawTweet['text'].lower().split()]
                    test_tweets.append(tweet)
                    neg_tweets+=1
            elif (rawTweet['relevant']==True):
                if (pos_tweets < 250):
                    tweet = [1,rawTweet['text'].lower().split()]
                    tweets.append(tweet)
                    pos_tweets+=1
                else:
                    tweet = [1,rawTweet['text'].lower().split()]
                    test_tweets.append(tweet)
                    pos_tweets+=1
            else:
                break


# In[3]:  Exporting the tweets that had been separated into training and test data.


for tweet in tweets:
    temp = [tweet[0],' '.join(tweet[1])]
    jtwt = json.dumps(temp)
    f = open('labeled_tweets.txt', 'a+')
    f.write(jtwt + '\n')
    f.close()

for tweet in test_tweets:
    jtwt = json.dumps(' '.join(tweet[1]))
    f = open('unlabeled_tweets.txt', 'a+')
    f.write(jtwt + '\n')
    f.close()


# In[4]:  Words not to consider as features


stopwords = stopwords = ["a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the", "https", "CO", "Lz3PqqUkWC"]


# In[5]: Counting the number of occurences of terms in all training tweets:


vocab = dict()
for class_label, text in tweets:
    for term in text:
        if (term not in stopwords) & (len(term) > 2):
            if term in vocab.keys():
                vocab[term] = vocab[term] + 1
            else:
                vocab[term] = 1


# In[6]:  Sort the terms by frequency:


sorted_vocab = sorted(vocab.items(), key=lambda kv: kv[1],reverse = True)


# In[7]:  Pick the top 10 terms:


topVocab = sorted_vocab[0:10]
vocab = dict()

for word in topVocab:
    vocab[word[0]] = word[1]


# In[8]:

# In[9]:

print("\n\n\n\n\n *************************************** \n\n\n\n")

print("The following is creating an SVM model using frequency of terms in a set of tweets to classify whether they are about traffic or not.\n\n\n")
# Generate an id (starting from 0) for each term in vocab
vocab = {term: idx for idx, (term, freq) in enumerate(vocab.items())}
print("The number of keywords used for generating features (frequencies): ", len(vocab))


# In[10]:


print('Features:',np.matrix(topVocab))
print('\n\n\n')

# In[11]:  Creating our feature matrix and label matrix of our training data:


X = []
y = []
for class_label, text in tweets:
    x = [0] * len(vocab)
    terms = [term for term in text]
    for term in terms:
        if term in vocab.keys():
            x[vocab[term]] += 1
    y.append(class_label)
    X.append(x)


# In[12]:


print ("The total number of training tweets: {} ({} positives, {}: negatives)".format(len(y), sum(y), len(y) - sum(y)))
print('\n\n\n')

# In[13]:
print('Training Feature Matrix:')
for row in range(np.size(X,axis=0)):
	print(X[row])
print('\n\n Training Label Matrix:')
print(np.reshape(y,(500,1)))


# In[14]:


tweets = []
for row in range(len(test_tweets)):
    toAppend = test_tweets[row][1]
    tweets = np.concatenate((tweets,toAppend),axis=0)


# In[15]:


# 10 folder cross validation to estimate the best w and b
svc = svm.SVC(kernel='linear')
grid = dict(C=list(range(1,20)))
clf = GridSearchCV(estimator=svc,param_grid=grid, cv = 10)
clf.fit(X, y)
print('\n\n Model Statistics: \n')
print ("The estimated w: ")
print (clf.best_estimator_.coef_)

print ("\nThe estimated b: ")
print (clf.best_estimator_.intercept_)

print ("\nThe estimated C after the grid search for 10 fold cross validation: ")
print (clf.best_params_)

print("\nAccuracy on the training data:")
print(clf.score(X,y))



# Generate feature matrix for test tweets
test_X = []
for text in test_tweets:
    x = [0] * len(vocab)
    for term in text[1]:
        if term in vocab.keys():
            x[vocab[term]] += 1
    test_X.append(x)
# predict the class labels of new tweets
test_y = clf.predict(test_X)

print ("\nThe total number of testing tweets: {} ({} are predicted as positives, {} are predicted as negatives)".format(len(test_y), sum(test_y), len(test_y) - sum(test_y)))


# In[16]:


true_positive = 0
false_positive = 0
true_negative = 0
false_negative = 0

#Determine tp,tn,fp,tn

for row in range(len(test_tweets)):
    if (test_tweets[row][0] == 1):
        if (test_y[row] ==1):
            true_positive += 1
        else:
            false_negative += 1
    else:
        if (test_y[row] ==0):
            true_negative += 1
        else:
            false_positive += 1
			
accuracy = (true_positive+true_negative)/len(test_y)
print("\nAccuracy on test data: ",accuracy)
print("True positives =",true_positive)
print("True negatives =",true_negative)
print("False Positives =",false_positive)
print("False Negatives =",false_negative)


# In[17]:





# In[18]:


coef = np.reshape(clf.best_estimator_.coef_,(np.size(clf.best_estimator_.coef_),(1)))


# In[19]:


featureNames = list(vocab.keys())
featureNames = np.reshape(featureNames,(np.size(featureNames),1))


# In[20]:


coef_names = np.concatenate((coef,featureNames),axis=1)


# In[21]:


print('\nCoefficients:')
print(coef_names)


# In[22]:


predicted_tweets = []
for row in range(len(test_y)):
    predicted_tweets.append([test_y[row],' '.join(test_tweets[row][1])])
    


# In[23]:


for tweet in predicted_tweets:
    tweet[0] = int(tweet[0])
    jtwt = json.dumps(tweet)
    f = open('predicted_tweets.txt', 'a+')
    f.write(jtwt + '\n')
    f.close()

