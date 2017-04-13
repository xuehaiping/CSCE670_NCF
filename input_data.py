
# coding: utf-8

# In[12]:

from keras.layers import Input, Dense, Activation
from keras.models import Model
import numpy as np


# In[46]:

#create index dictionary for an list
def idx_dict(array):
    idx_dict = {}
    for i in range(0,len(array)):
        idx_dict[array[i]] = i
    return idx_dict

#build interaction matrix
def interaction_matrix(u_dict, movs, usrs): 
    interaction_vector = np.zeros((len(usrs), len(movs)))
    #create idx dictionary for movie and user
    movie_idx_dict = idx_dict(movs)
    user_idx_dict = idx_dict(usrs)
    for usr in usrs:
        for mov in list(u_dict[usr]):
             interaction_vector[user_idx_dict[usr]][movie_idx_dict[mov]] = 1
    return interaction_vector

#create one hot vector for a list 
def one_hot_vector(array):
    one_vector = np.zeros((len(array), len(array)))
    for i in range(len(array)):
        one_vector[i][i] = 1
    return one_vector


# In[47]:

##build user dictionary, user list can be created by getting the keys for user dictionary 
user_dict = {}
#set for movies
movies = set()
#user_list
users = []

# read data from ratings.csv, userId, movieId, timestamp
file_path = 'ratings.csv'
##create user dictionary
with open(file_path, 'rb') as f:
    f.readline()
    for i in range(1,40000):
        data = f.readline().split(",")
        #add movie to movie list
        if int(data[1]) not in movies:
            movies.add(int(data[1]))   
        #add data into dictionary
        if int(data[0]) in user_dict:
            user_dict[int(data[0])].add(int(data[1]))
        else:
            user_dict[int(data[0])] = set()
            user_dict[int(data[0])].add(int(data[1]))
f.close()

#convert it back to list
movies = list(movies)
#add users Id from use dictionary 
users = user_dict.keys()

#interaction matrix 
interact_mx = interaction_matrix(user_dict, movies, users)
#create one hot matrix for user
one_hot_user = one_hot_vector(users)
#create one hot matrix for movies
one_hot_user = one_hot_vector(movies)


# In[ ]:







