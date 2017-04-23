from keras.models import Model
from keras.layers import Dense, Activation,Embedding,Input,concatenate,multiply,Flatten
import numpy as np
import ncf_helper as helper
import keras.layers as layers
from keras import initializers


num_predictive_factors = 8
batch_size = 2
# embedding size is 2 * num_predictive_factors if MLP is 3 layered

# load data
inputs, labels = helper.training_data_generation('input/training_data.npy','int_mat.npy',5)
interaction_mx = np.load('input/int_mat.npy')

#https://datascience.stackexchange.com/questions/13428/what-is-the-significance-of-model-merging-in-keras
user_input = Input(shape=(1,),name='user_input')
item_input = Input(shape=(1,),name='item_input')

user_embed = Flatten()(Embedding(interaction_mx.shape[0] + 1,
                                 num_predictive_factors * 2,
                                 #W_regularizer = l2(0.01),
                                 input_length=1,
                                 #dropout = 0.3,
                                 embeddings_initializer = initializers.RandomNormal(mean = 0.0, stddev=0.01, seed=None))(user_input))
item_embed = Flatten()(Embedding(interaction_mx.shape[1] + 1,
                                 num_predictive_factors * 2,
                                 #W_regularizer = l2(0.01),
                                 input_length=1,
                                 #dropout = 0.3,
                                 embeddings_initializer = initializers.RandomNormal(mean = 0.0, stddev=0.01, seed=None))(item_input))
merged_embed = multiply([user_embed, item_embed])

main_output = Dense(1, activation='sigmoid',name='main_output')(merged_embed)

model = Model(inputs=[user_input, item_input], output=main_output)
model.compile(optimizer='Adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(inputs, labels, batch_size = 256, epochs = 2)

user_embed_weights = model.get_layer('user_embed').get_weights()
item_embed_weights = model.get_layer('item_embed').get_weights()
main_output_weights = model.get_layer('main_output').get_weights()

np.save('GMF_WE/GMF_user_embed.npy', user_embed_weights)
np.save('GMF_WE/GMF_item_embed.npy', item_embed_weights)
np.save('GMF_WE/GMF_output_layer.npy', main_output_weights)