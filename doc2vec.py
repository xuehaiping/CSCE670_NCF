
# coding: utf-8

# # Doc2Vec
# 

# # Data: Yelp review data
# 
# In this assignment, given a Yelp review, your task is to implement two classifiers to predict if the business category of this review is "food-relevant" or not, **only based on the review text**. The data is from the [Yelp Dataset Challenge](https://www.yelp.com/dataset_challenge).
# 
# ## Build the training data
# 
# First, you will need to download this data file as your training data: [training_data.json](https://drive.google.com/open?id=0B_13wIEAmbQMdzBVTndwenoxQlk) 
# 
# The training data file includes 40,000 Yelp reviews. Each line is a json-encoded review, and **you should only focus on the "text" field**. I removed stop words and did casefolding and stemming**.
# 
# The label (class) information of each review is in the "label" field. It is **either "Food-relevant" or "Food-irrelevant"**.
# 
# ## Testing data
# 
# We provide 100 yelp reviews here: [testing_data.json](https://drive.google.com/open?id=0B_13wIEAmbQMbXdyTkhrZDN4Wms). The testing data file has the same format as the training data file. Again, you can get the label informaiton in the "label" field. Only use it when you evalute your classifiers.

# In[1]:

# read data method
import time
start_time = time.time()



def read_data(file_path):
    
    import json
    
    # read json data
    json_list = []
    with open(file_path) as json_data:
        for every_object in json_data:
            json_decode=json.loads(every_object)
            json_list.append(json_decode)
            
            
    #read review data and relevant data
    review_list = []
    relevant_list = []
    
    for item in json_list:
        review = item['text']
        
        relevant = item['stars']  # "stars": 1-5
        
        review_list.append(review)
        relevant_list.append(relevant)
        
    return review_list[0:1000], relevant_list[0:1000]


# In[2]:

# process data method
# input a list of strings
# output a list of word lists

def process_data(str_list):
    import re
    import nltk
    from nltk.stem.porter import PorterStemmer
    from nltk.corpus import stopwords
    review_word_list = []
    stopWords = set(stopwords.words('english'))
    
    for item in str_list:
        
        word_list = re.findall(r"\w+",item)
        word_list = [w.lower() for w in word_list]
        
        word_list_filter = []
        for w in word_list:
            if w not in stopWords:
                word_list_filter.append(w)
        
        stemmer = PorterStemmer()
        word_list = [stemmer.stem(w) for w in word_list_filter]
        review_word_list.append(word_list)
        
    return review_word_list


# In[3]:

from nltk.corpus import stopwords
stopWords = set(stopwords.words('english'))
#==============================================================================
# print(stopWords)
#==============================================================================


# In[4]:

# read trainning data
path_training = "./training_data.json"

review_list_training, relevant_list_training = read_data(path_training)

# print review_list_training[0]


# In[5]:

#==============================================================================
# print relevant_list_training
# print review_list_training[0]
#==============================================================================


# In[6]:

# process trainning data
review_word_list_training = process_data(review_list_training)

# len(review_word_list_training)


# In[7]:

#==============================================================================
# print review_word_list_training[0]
#==============================================================================


# In[8]:

# read testing data
path_testing = "./testing_data.json"

review_list_testing, relevant_list_testing = read_data(path_testing)

# print review_list_testing[0]


# In[9]:

# process testing data
review_word_list_testing = process_data(review_list_testing)

len(review_word_list_testing)

# print review_word_list_testing[0]


# In[10]:

import gensim

LabeledSentence = gensim.models.doc2vec.LabeledSentence

import numpy as np


# In[11]:

x_train = review_word_list_training
# x_train = review_list_training
x_test = review_word_list_testing

y_train = relevant_list_training
# y_train = review_list_testing
y_test = relevant_list_testing


print len(x_train)
print len(x_test)

# In[12]:

#==============================================================================
# print x_train[0], x_test[0], y_train, y_test
#==============================================================================


# In[13]:

#Do some very minor text preprocessing
def cleanText(corpus):
    punctuation = """.,?!:;(){}[]"""
    corpus = [z.lower().replace('\n','') for z in corpus]
    corpus = [z.replace('<br />', ' ') for z in corpus]

    #treat punctuation as individual words
    for c in punctuation:
        corpus = [z.replace(c, ' %s '%c) for z in corpus]
    corpus = [z.split() for z in corpus]
    return corpus

x_train = review_list_training
x_test = review_list_testing

x_train = cleanText(x_train)
x_test = cleanText(x_test)


# In[14]:

#==============================================================================
# print x_train[0]
#==============================================================================


# In[15]:

# x_train = cleanText(x_train)
# x_test = cleanText(x_test)


# In[16]:

#Gensim's Doc2Vec implementation requires each document/paragraph to have a label associated with it.
#We do this by using the LabeledSentence method. The format will be "TRAIN_i" or "TEST_i" where "i" is
#a dummy index of the review.
def labelizeReviews(reviews, label_type):
    labelized = []
    for i,v in enumerate(reviews):
        label = '%s_%s'%(label_type,i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized


# In[17]:

x_train = labelizeReviews(x_train, 'TRAIN')
x_test = labelizeReviews(x_test, 'TEST')
# unsup_reviews = labelizeReviews(unsup_reviews, 'UNSUP')


# In[18]:

#==============================================================================
# print x_train[0]
#==============================================================================
np.concatenate((x_train, x_test))


# In[ ]:

import random

size = 400

#instantiate our DM and DBOW models
model_dm = gensim.models.Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, workers=3)
model_dbow = gensim.models.Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, dm=0, workers=3)

#build vocab over all reviews
model_dm.build_vocab(x_train + x_test)
model_dbow.build_vocab(x_train + x_test)


#We pass through the data set multiple times, shuffling the training reviews each time to improve accuracy.
# all_train_reviews = np.concatenate((x_train, unsup_reviews))
all_train_reviews = x_train
for epoch in range(10):
    random.shuffle(all_train_reviews)
    model_dm.train(all_train_reviews)
    model_dbow.train(all_train_reviews)

#Get training set vectors from our models
def getVecs(model, corpus, size):
    vecs = [np.array(model.docvecs[z.tags[0]]).reshape((1, size)) for z in corpus]
    return np.concatenate(vecs)

train_vecs_dm = getVecs(model_dm, x_train, size)
train_vecs_dbow = getVecs(model_dbow, x_train, size)

train_vecs = np.hstack((train_vecs_dm, train_vecs_dbow))

#train over test set


for epoch in range(10):
    random.shuffle(x_test)
    model_dm.train(x_test)
    model_dbow.train(x_test)

#Construct vectors for test reviews
test_vecs_dm = getVecs(model_dm, x_test, size)
test_vecs_dbow = getVecs(model_dbow, x_test, size)

test_vecs = np.hstack((test_vecs_dm, test_vecs_dbow))


# In[ ]:

from sklearn.linear_model import SGDClassifier

lr = SGDClassifier(loss='log', penalty='l1')
lr.fit(train_vecs, y_train)

print 'Test Accuracy: %.2f'%lr.score(test_vecs, y_test)


print("--- %s seconds ---" % (time.time() - start_time))