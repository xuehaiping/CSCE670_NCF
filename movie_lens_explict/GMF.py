from keras.models import Model
from keras.layers import Dense, Activation, Embedding, Input, concatenate, multiply, Flatten
import numpy as np
import evaluation
import data_management
import keras.layers as layers
from keras import initializers
import keras

def create_model(num_users, num_items, num_predictive_factors, pretrain):

    user_input = Input(shape=(1,), name='user_input')
    item_input = Input(shape=(1,), name='item_input')
    user_embed = Flatten()(Embedding(num_users,
                                     num_predictive_factors * 2,
                                     # W_regularizer = l2(0.01),
                                     input_length=1,
                                     # dropout = 0.3,
                                     name='GMF_user_embed',
                                     embeddings_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01,
                                                                                      seed=None))(user_input))
    item_embed = Flatten()(Embedding(num_items,
                                     num_predictive_factors * 2,
                                     # W_regularizer = l2(0.01),
                                     input_length=1,
                                     # dropout = 0.3,
                                     name='GMF_item_embed',
                                     embeddings_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01,
                                                                                      seed=None))(item_input))
    GMF_merged_embed = multiply([user_embed, item_embed])
    main_output = Dense(6, activation='softmax', name='main_output')(GMF_merged_embed)

    if pretrain:
        model = Model(inputs=[user_input, item_input], output=main_output)
    else:
        model = Model(inputs=[user_input, item_input], output=GMF_merged_embed)
    return model


def train_gmf(num_predictive_factors, batch_size, epochs, interaction_mx, inputs, labels):
    pretrain_model = create_model(num_users=interaction_mx.shape[0],
                                  num_items=interaction_mx.shape[1],
                                  num_predictive_factors=num_predictive_factors,
                                  pretrain=True)

    pretrain_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    pretrain_model.fit(inputs, labels, batch_size, epochs)

    user_embed_weights = pretrain_model.get_layer('GMF_user_embed').get_weights()
    item_embed_weights = pretrain_model.get_layer('GMF_item_embed').get_weights()
    main_output_weights = pretrain_model.get_layer('main_output').get_weights()

    np.save('GMF_WE/GMF_user_embed.npy', user_embed_weights)
    np.save('GMF_WE/GMF_item_embed.npy', item_embed_weights)
    np.save('GMF_WE/GMF_output_layer.npy', main_output_weights)

    #hit_rate_accuracy = evaluation.evaluate_integer_input('input/testing_data.npy', pretrain_model, 'hit_rate',
    #                                                      'input/int_mat.npy')
    #print('accuracy rate of: ' + str(hit_rate_accuracy))
#
#
# if __name__ == '__main__':
#     try:
#         interaction_mx = np.load('input/int_mat.npy')
#     except IOError:
#         data_management.load_data()
#         interaction_mx = np.load('input/int_mat.npy')
#     inputs, labels = data_management.training_data_generation('input/training_data.npy', 'input/int_mat.npy', 5)
#     train_gmf(num_predictive_factors=8, batch_size=256, epochs=2,
#               interaction_mx=interaction_mx, inputs=inputs, labels=labels)
