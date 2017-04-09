from keras.models import Model
from keras.layers import Dense, Activation,Embedding,Input,concatenate
import numpy as np
import keras.layers as layers

num_predictive_factors = 8
# embedding size is 2 * num_predictive_factors if MLP is 3 layered

# generate fake data
num_users = 20
num_items = 30
dummy_user_data = np.zeros(num_users)
dummy_item_data = np.zeros(num_items)
dummy_item_data[np.random.randint(len(dummy_item_data) + 1)] = 1
dummy_user_data[np.random.randint(len(dummy_user_data) + 1)] = 1

#https://datascience.stackexchange.com/questions/13428/what-is-the-significance-of-model-merging-in-keras
user_input = Input(shape=(num_users,))
item_input = Input(shape=(num_items,))
user_embed = Embedding(2, num_predictive_factors * 2, input_length=num_users)(user_input)
item_embed = Embedding(2, num_predictive_factors * 2, input_length=num_items)(item_input)
merged_embed = concatenate([user_embed, item_embed], axis=1)
mlp_1 = Dense(32, activation='relu')(merged_embed)
mlp_2 = Dense(16, activation='relu')(mlp_1)
mlp_3 = Dense(8, activation='relu')(mlp_2)
main_output = Dense(1, activation='relu')(mlp_3)
model = Model(inputs=[user_input, item_input], output = merged_embed)
model.compile(optimizer='Adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

#user_mlp = Sequential()
#item_mlp = Sequential()

#user_mlp.add(Embedding(2, num_predictive_factors * 2, input_length=num_users))
#item_mlp.add(Embedding(2, num_predictive_factors * 2, input_length=num_items))
#https://github.com/fchollet/keras/issues/3921
#merged_embed = Merge([user_mlp, item_mlp], mode='concat', concat_axis=1)
#concat_layer = Merge.Concatenate(axis=1)
#merged_embed = concat_layer([user_mlp, item_mlp])