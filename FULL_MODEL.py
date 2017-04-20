
# coding: utf-8

# In[6]:

from keras.models import Model
from keras.layers import Dense, Activation,Embedding,Input,concatenate, multiply
import numpy as np
import keras.layers as layers

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


def load_weights(model):
    model.get_layer('MLP_user_embed').set_weights(np.load('MLP_WE/mlp_user_embed_weights.npy'))
    model.get_layer('MLP_item_embed').set_weights(np.load('MLP_WE/mlp_item_embed_weights.npy'))
    
    mlp1_0 = np.load('MLP_WE/mlp_1_weights_array0.npy')
    mlp1_1 = np.load('MLP_WE/mlp_1_weights_array1.npy')    
    model.get_layer('mlp_1').set_weights([mlp1_0,mlp1_1])
    
    model.get_layer('mlp_2').set_weights(np.load('MLP_WE/mlp_2_weights.npy'))
    model.get_layer('mlp_3').set_weights(np.load('MLP_WE/mlp_3_weights.npy'))

    model.get_layer('GMF_user_embed').set_weights(np.load('GMF_WE/GMF_user_embed.npy'))
    model.get_layer('GMF_item_embed').set_weights(np.load('GMF_WE/GMF_item_embed.npy'))
    model.get_layer('GMF_layer').set_weights(np.load('GMF_WE/GMF_output_layer.npy'))
    
    return model


num_predictive_factors = 8
batch_size = 1
one_hot_users = np.load('input/one_hot_user.npy')
one_hot_movies = np.load('input/one_hot_movies.npy')
interaction_mx = np.load('input/interaction_mx.npy')

#----- MLP Model -----
user_input = Input(shape=(len(one_hot_users),),name='user_input')
item_input = Input(shape=(len(one_hot_movies),),name='item_input')

MLP_user_embed = Dense(num_predictive_factors * 2, activation='sigmoid',name='MLP_user_embed')(user_input)
MLP_item_embed = Dense(num_predictive_factors * 2, activation='sigmoid',name='MLP_item_embed')(item_input)

MLP_merged_embed = concatenate([MLP_user_embed, MLP_item_embed], axis=1)
mlp_1 = Dense(32, activation='relu',name='mlp_1')(MLP_merged_embed)
mlp_2 = Dense(16, activation='relu',name='mlp_2')(mlp_1)
mlp_3 = Dense(8, activation='relu',name='mlp_3')(mlp_2) #This will be the input for the final layer
MLP_main_output = Dense(1, activation='sigmoid',name='MLP_main_output')(mlp_3)

#----- GMF Model -----
GMF_user_embed = Dense(num_predictive_factors * 2, activation='sigmoid', name='GMF_user_embed')(user_input)
GMF_item_embed = Dense(num_predictive_factors * 2, activation='sigmoid', name='GMF_item_embed')(item_input)

GMF_merged_embed = multiply([GMF_user_embed, GMF_item_embed])

GMF_layer = Dense(1, activation='sigmoid',name='GMF_layer')(GMF_merged_embed)

#Concatenate with GMF last layer
#MLP_input = Input(shape=(len(one_hot_users),),name='MLP_input') #This may be necessary
gmf_mlp_concatenated = concatenate([MLP_main_output, GMF_layer], axis=1);


#Feed previous concatenate to NeuMF Layer
NeuMF = Dense(2, activation='relu', name='NeuMF')(gmf_mlp_concatenated)
NeuMF_main_output = Dense(1, activation='sigmoid',name='NeuMF_main_output')(NeuMF)

model = Model(inputs=[user_input, item_input], output=NeuMF_main_output)

model = load_weights(model)

model.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=['accuracy'])


