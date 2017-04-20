from keras.models import Model
from keras.layers import Dense, Activation,Embedding,Input,concatenate,multiply
import numpy as np
import keras.layers as layers



def preprocess_data(users_matrix, items_matrix, interactions_matrix, batch_size):
    if (interactions_matrix.size - len(users_matrix[0])) % batch_size != 0:
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

                    


num_predictive_factors = 8
batch_size = 2
# embedding size is 2 * num_predictive_factors if MLP is 3 layered

# load data
one_hot_users = np.load('one_hot_user.npy')
one_hot_movies = np.load('one_hot_movies.npy')
interaction_mx = np.load('interaction_mx.npy')

#users, items, interactions = preprocess_data(one_hot_users,one_hot_movies,interaction_mx)


#https://datascience.stackexchange.com/questions/13428/what-is-the-significance-of-model-merging-in-keras
user_input = Input(shape=(len(one_hot_users),),name='user_input')
item_input = Input(shape=(len(one_hot_movies),),name='item_input')

user_embed = Dense(num_predictive_factors * 2, activation='sigmoid', name = 'user_embed')(user_input)
item_embed = Dense(num_predictive_factors * 2, activation='sigmoid', name = 'item_embed')(item_input)
merged_embed = multiply([user_embed, item_embed])

main_output = Dense(1, activation='sigmoid',name='main_output')(merged_embed)

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
#                       steps = 1)

user_embed_weights = model.get_layer('user_embed').get_weights()
item_embed_weights = model.get_layer('item_embed').get_weights()
main_output_weights = model.get_layer('main_output').get_weights()

np.save('GMF_user_embed.npy', user_embed_weights)
np.save('GMF_item_embed.npy', item_embed_weights)
np.save('GMF_output_layer.npy', main_output_weights)
