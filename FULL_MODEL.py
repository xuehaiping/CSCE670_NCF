
# coding: utf-8

# In[1]:

from keras.models import Model
from keras.layers import Dense, Activation,Embedding,Input,concatenate
import numpy as np
import keras.layers as layers


# In[2]:

def preprocess_data(users_matrix, items_matrix, interactions_matrix, batch_size):
    if (interactions_matrix.size % batch_size) != 0:
        print(str(interactions_matrix.size - len(users_matrix[0])) + 'is not divisible by ' + str(batch_size))
        raise StandardError
        
    users = []
    items = []
    interactions = []
    while True:
        for user_idx, user in enumerate(users_matrix):
            for item_idx, item in enumerate(items_matrix):
                if interactions_matrix[user_idx][item_idx] != -1:
                    users.append(user)
                    items.append(item)
                    interactions.append(interactions_matrix[user_idx][item_idx])
                if len(users) == batch_size:
                    yield ({'user_input': np.array(users), 'item_input': np.array(items)},
                           np.array(interactions))
                    users = []
                    items = []
                    interactions = []


# In[ ]:

#TODO: Load pretrained weights


# In[3]:

num_predictive_factors = 8
batch_size = 1
one_hot_users = np.load('one_hot_user.npy')
one_hot_movies = np.load('one_hot_movies.npy')
interaction_mx = np.load('interaction_mx.npy')


# In[6]:

#TODO: Feed layers with pretrained weights for MLP and GMF

#----- MLP Model -----
MLP_user_input = Input(shape=(len(one_hot_users),),name='MLP_user_input')
MLP_item_input = Input(shape=(len(one_hot_movies),),name='MLP_item_input')

MLP_user_embed = Dense(num_predictive_factors * 2, activation='sigmoid',name='MLP_user_embed')(MLP_user_input)
MLP_item_embed = Dense(num_predictive_factors * 2, activation='sigmoid',name='MLP_item_embed')(MLP_item_input)

MLP_merged_embed = concatenate([MLP_user_embed, MLP_item_embed], axis=1)
mlp_1 = Dense(32, activation='relu',name='mlp_1')(MLP_merged_embed)
mlp_2 = Dense(16, activation='relu',name='mlp_2')(mlp_1)
mlp_3 = Dense(8, activation='relu',name='mlp_3')(mlp_2) #This will be the input for the final layer

#----- GMF Model -----
#TODO: Concatenate with GMF last layer
#gmf_mlp_concatenated = concatenate([mlp_3, mlp_last_layer], axis=1);


#TODO: Feed previous concatenate to NeuMF Layer


# In[ ]:



