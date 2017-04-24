import MLP, GMF, data_management, evaluation
from keras.models import Model
from keras.layers import Dense, Embedding, Input, concatenate, multiply, Flatten
import numpy as np

data_management.load_data(file_path='data/movielens/ratings.dat')
# pretrain MLP
MLP.train_mlp(num_predictive_factors=8, batch_size=256, epochs=2)
# pretrain GMF
GMF.train_gmf(num_predictive_factors=8, batch_size=256, epochs=2)


def load_weights(model):
    model.get_layer('MLP_user_embed').set_weights(np.load('MLP_WE/mlp_user_embed_weights.npy'))
    model.get_layer('MLP_item_embed').set_weights(np.load('MLP_WE/mlp_item_embed_weights.npy'))

    mlp1_0 = np.load('MLP_WE/mlp_1_weights_array0.npy')
    mlp1_1 = np.load('MLP_WE/mlp_1_weights_array1.npy')
    model.get_layer('mlp_1').set_weights([mlp1_0, mlp1_1])

    model.get_layer('mlp_2').set_weights(np.load('MLP_WE/mlp_2_weights.npy'))
    model.get_layer('mlp_3').set_weights(np.load('MLP_WE/mlp_3_weights.npy'))

    model.get_layer('GMF_user_embed').set_weights(np.load('GMF_WE/GMF_user_embed.npy'))
    model.get_layer('GMF_item_embed').set_weights(np.load('GMF_WE/GMF_item_embed.npy'))
    # model.get_layer('GMF_main_output').set_weights(np.load('GMF_WE/GMF_output_layer.npy'))

    return model


num_predictive_factors = 8
batch_size = 256

# ----- MLP Model -----
interaction_mx = np.load('input/int_mat.npy')
# load data
inputs, labels = data_management.training_data_generation('input/training_data.npy', 'input/int_mat.npy', 5)
user_input = Input(shape=(1,), name='user_input')
item_input = Input(shape=(1,), name='item_input')
MLP_user_embed = Flatten()(Embedding(interaction_mx.shape[0] + 1,
                                     num_predictive_factors * 2,
                                     # W_regularizer = l2(0.01),
                                     input_length=1,
                                     # dropout = 0.3,
                                     name='MLP_user_embed'
                                     # embeddings_initializer = initializers.RandomNormal(mean = 0.0, stddev=0.01, seed=None)
                                     )(user_input))
MLP_item_embed = Flatten()(Embedding(interaction_mx.shape[1] + 1,
                                     num_predictive_factors * 2,
                                     # W_regularizer = l2(0.01),
                                     input_length=1,
                                     # dropout = 0.3,
                                     name='MLP_item_embed'
                                     # embeddings_initializer = initializers.RandomNormal(mean = 0.0, stddev=0.01, seed=None)
                                     )(item_input))
MLP_merged_embed = concatenate([MLP_user_embed, MLP_item_embed], axis=1)
mlp_1 = Dense(32, activation='relu',
              # W_regularizer = l2(0.01),
              name='mlp_1')(MLP_merged_embed)
mlp_2 = Dense(16, activation='relu',
              # W_regularizer = l2(0.01),
              name='mlp_2')(mlp_1)
mlp_3 = Dense(8, activation='relu',
              # W_regularizer = l2(0.01),
              name='mlp_3')(mlp_2)

# ----- GMF Model -----

GMF_user_embed = Flatten()(Embedding(interaction_mx.shape[0] + 1,
                                     num_predictive_factors * 2,
                                     # W_regularizer = l2(0.01),
                                     input_length=1,
                                     # dropout = 0.3,
                                     name='GMF_user_embed'
                                     # embeddings_initializer = initializers.RandomNormal(mean = 0.0, stddev=0.01, seed=None)
                                     )(user_input))
GMF_item_embed = Flatten()(Embedding(interaction_mx.shape[1] + 1,
                                     num_predictive_factors * 2,
                                     # W_regularizer = l2(0.01),
                                     input_length=1,
                                     # dropout = 0.3,
                                     name='GMF_item_embed'
                                     # embeddings_initializer = initializers.RandomNormal(mean = 0.0, stddev=0.01, seed=None)
                                     )(item_input))
GMF_merged_embed = multiply([GMF_user_embed, GMF_item_embed])

# Concatenate with GMF last layer
# MLP_input = Input(shape=(len(one_hot_users),),name='MLP_input') #This may be necessary
gmf_mlp_concatenated = concatenate([mlp_3, GMF_merged_embed], axis=1)

# Feed previous concatenate to NeuMF Layer
NeuMF = Dense(16, activation='sigmoid', name='NeuMF')(gmf_mlp_concatenated)
NeuMF_main_output = Dense(1, activation='sigmoid', name='NeuMF_main_output')(NeuMF)

model = Model(inputs=[user_input, item_input], output=NeuMF_main_output)

model = load_weights(model)

model.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(inputs, labels, batch_size=256, epochs=10)
hit_rate_accuracy = evaluation.evaluate_integer_input('input/testing_data.npy', model, 'hit_rate', 'input/int_mat.npy')
print('accuracy rate of: ' + str(hit_rate_accuracy))
model.save('final_model.h5')
