from keras.models import Model
from keras.layers import Dense, Activation,Embedding,Input,concatenate,multiply,Flatten
import numpy as np
import evaluation
import data_management
import keras.layers as layers
from keras import initializers


def train_gmf(num_predictive_factors, batch_size, epochs):
    # embedding size is 2 * num_predictive_factors if MLP is 3 layered

    # load data
    inputs, labels = data_management.training_data_generation('input/training_data.npy', 'input/int_mat.npy', 5)
    interaction_mx = np.load('input/int_mat.npy')

    user_input = Input(shape=(1,),name='user_input')
    item_input = Input(shape=(1,),name='item_input')

    user_embed = Flatten()(Embedding(interaction_mx.shape[0] + 1,
                                     num_predictive_factors * 2,
                                     #W_regularizer = l2(0.01),
                                     input_length=1,
                                     #dropout = 0.3,
                                     name='user_embed',
                                     embeddings_initializer = initializers.RandomNormal(mean = 0.0, stddev=0.01, seed=None))(user_input))
    item_embed = Flatten()(Embedding(interaction_mx.shape[1] + 1,
                                     num_predictive_factors * 2,
                                     #W_regularizer = l2(0.01),
                                     input_length=1,
                                     #dropout = 0.3,
                                     name='item_embed',
                                     embeddings_initializer = initializers.RandomNormal(mean = 0.0, stddev=0.01, seed=None))(item_input))
    merged_embed = multiply([user_embed, item_embed])

    main_output = Dense(1, activation='sigmoid',name='main_output')(merged_embed)

    model = Model(inputs=[user_input, item_input], output=main_output)
    model.compile(optimizer='Adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(inputs, labels, batch_size, epochs)

    user_embed_weights = model.get_layer('user_embed').get_weights()
    item_embed_weights = model.get_layer('item_embed').get_weights()
    main_output_weights = model.get_layer('main_output').get_weights()

    np.save('GMF_WE/GMF_user_embed.npy', user_embed_weights)
    np.save('GMF_WE/GMF_item_embed.npy', item_embed_weights)
    np.save('GMF_WE/GMF_output_layer.npy', main_output_weights)

    hit_rate_accuracy = evaluation.evaluate_integer_input('input/testing_data.npy', model, 'hit_rate', 'input/int_mat.npy')
    print('accuracy rate of: ' + str(hit_rate_accuracy))

if __name__ == '__main__':
    data_management.load_data(file_path='data/movielens/ratings.dat')
    train_gmf(num_predictive_factors=8, batch_size=256, epochs=2)
