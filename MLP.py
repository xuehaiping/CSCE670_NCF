from keras.models import Model
from keras.layers import Dense, Activation,Embedding,Input,concatenate
import numpy as np
import keras.layers as layers

def preprocess_data(users_matrix, items_matrix, interactions_matrix, batch_size):
    if interactions_matrix.size % batch_size != 0:
        print(str(interactions_matrix.size) + 'is not divisible by ' + str(batch_size))
        raise StandardError
    users = []
    items = []
    interactions = []
    while True:
        for user_idx, user in enumerate(users_matrix):
            for item_idx, item in enumerate(items_matrix):
                users.append(user)
                items.append(item)
                interactions.append(interactions_matrix[user_idx][item_idx])
                if len(users) == batch_size:
                    yield ({'user_input': np.array(users), 'item_input': np.array(items)},
                           np.array(interactions))
                    users = []
                    items = []
                    interactions = []


num_predictive_factors = 8
batch_size = 1
# embedding size is 2 * num_predictive_factors if MLP is 3 layered

# load data
one_hot_users = np.load('one_hot_user.npy')
one_hot_movies = np.load('one_hot_movies.npy')
interaction_mx = np.load('interaction_mx.npy')

#users, items, interactions = preprocess_data(one_hot_users,one_hot_movies,interaction_mx)


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
#model.fit([users, items], interactions,epochs=50, batch_size=32)

model.fit_generator(preprocess_data(one_hot_users,one_hot_movies,interaction_mx, batch_size),
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