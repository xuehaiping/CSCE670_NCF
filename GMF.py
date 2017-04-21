from keras.models import Model
from keras.layers import Dense, Activation,Embedding,Input,concatenate,multiply,Flatten
import numpy as np
import ncf_helper as helper
import keras.layers as layers

num_predictive_factors = 8
batch_size = 2
# embedding size is 2 * num_predictive_factors if MLP is 3 layered

# load data
inputs, labels = helper.training_data_generation('input/one_training_data')


#https://datascience.stackexchange.com/questions/13428/what-is-the-significance-of-model-merging-in-keras
user_input = Input(shape=(1,),name='user_input')
item_input = Input(shape=(1,),name='item_input')

user_embed = Flatten()(Embedding(len(inputs['user_input']) + 1, num_predictive_factors * 2, input_length=1)(user_input))
item_embed = Flatten()(Embedding(len(inputs['item_input']) + 1, num_predictive_factors * 2, input_length=1)(item_input))
merged_embed = multiply([user_embed, item_embed])

main_output = Dense(1, activation='sigmoid',name='main_output')(merged_embed)

model = Model(inputs=[user_input, item_input], output=main_output)
model.compile(optimizer='Adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(inputs, labels, batch_size = 64, epochs = 10)

user_embed_weights = model.get_layer('user_embed').get_weights()
item_embed_weights = model.get_layer('item_embed').get_weights()
main_output_weights = model.get_layer('main_output').get_weights()

np.save('GMF_WE/GMF_user_embed.npy', user_embed_weights)
np.save('GMF_WE/GMF_item_embed.npy', item_embed_weights)
np.save('GMF_WE/GMF_output_layer.npy', main_output_weights)
