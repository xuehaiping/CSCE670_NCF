from keras.models import Model
from keras.layers import Dense, Input,concatenate
import numpy as np
import ncf_helper as helper

num_predictive_factors = 8

batch_size = 1

# embedding size is 2 * num_predictive_factors if MLP is 3 layered

# load data
one_hot_users = np.load('one_hot_user.npy')
one_hot_movies = np.load('one_hot_movies.npy')
interaction_mx = np.load('interaction_mx.npy')

#https://datascience.stackexchange.com/questions/13428/what-is-the-significance-of-model-merging-in-keras
user_input = Input(shape=(len(one_hot_users),),name='user_input')
item_input = Input(shape=(len(one_hot_movies),),name='item_input')

#user_embed = Embedding(2, num_predictive_factors * 2, input_length=len(one_hot_users))(user_input)
#item_embed = Embedding(2, num_predictive_factors * 2, input_length=len(one_hot_movies))(item_input)
user_embed = Dense(num_predictive_factors * 2, activation='sigmoid',name='MLP_user_embed')(user_input)
item_embed = Dense(num_predictive_factors * 2, activation='sigmoid',name='MLP_item_embed')(item_input)
merged_embed = concatenate([user_embed, item_embed], axis=1)
mlp_1 = Dense(32, activation='relu', name='mlp_1')(merged_embed)
mlp_2 = Dense(16, activation='relu', name='mlp_2')(mlp_1)
mlp_3 = Dense(8, activation='relu', name='mlp_3')(mlp_2)
main_output = Dense(1, activation='sigmoid',name='main_output')(mlp_3)

model = Model(inputs=[user_input, item_input], output=main_output)
model.compile(optimizer='Adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit_generator(helper.preprocess_data(one_hot_users,one_hot_movies,interaction_mx, batch_size),
                    steps_per_epoch=interaction_mx.size/batch_size,
                    epochs=10,
                    verbose=1)
#result = model.predict_generator(preprocess_data(one_hot_users,one_hot_movies,interaction_mx, batch_size),
#                        steps = 1)

#Save weights for full_model
#Save weights for full_model
wmlp1= model.get_layer('mlp_1').get_weights()
np.save('mlp_1_weights_array0', wmlp1[0])
np.save('mlp_1_weights_array1', wmlp1[1])
np.save('mlp_2_weights',model.get_layer('mlp_2').get_weights());
np.save('mlp_3_weights',model.get_layer('mlp_3').get_weights());
np.save('mlp_user_embed_weights',model.get_layer('MLP_user_embed').get_weights());
np.save('mlp_item_embed_weights',model.get_layer('MLP_item_embed').get_weights());

