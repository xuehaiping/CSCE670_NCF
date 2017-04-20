from keras.models import Model
from keras.layers import Dense, Input,concatenate
import numpy as np
import ncf_helper as helper

num_predictive_factors = 8
batch_size = 684
# embedding size is 2 * num_predictive_factors if MLP is 3 layered

# load data
one_hot_users = np.load('one_hot_user.npy')
one_hot_movies = np.load('one_hot_movies.npy')
interaction_mx = np.load('interaction_mx.npy')

#https://datascience.stackexchange.com/questions/13428/what-is-the-significance-of-model-merging-in-keras
user_input = Input(shape=(len(one_hot_users),),name='user_input')
item_input = Input(shape=(len(one_hot_movies),),name='item_input')
user_embed = Dense(num_predictive_factors * 2, activation='sigmoid')(user_input)
item_embed = Dense(num_predictive_factors * 2, activation='sigmoid')(item_input)
merged_embed = concatenate([user_embed, item_embed], axis=1)
mlp_1 = Dense(32, activation='relu')(merged_embed)
mlp_2 = Dense(16, activation='relu')(mlp_1)
mlp_3 = Dense(8, activation='relu')(mlp_2)
main_output = Dense(1, activation='sigmoid', name='main_output')(mlp_3)

model = Model(inputs=[user_input, item_input], output=main_output)
model.compile(optimizer='Adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit_generator(helper.preprocess_data(one_hot_users,one_hot_movies,interaction_mx, batch_size),
                    steps_per_epoch=interaction_mx.size/batch_size,
                    epochs=10,
                    verbose=1)