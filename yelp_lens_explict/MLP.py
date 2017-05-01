from keras.models import Model
from keras.layers import Dense, Input,concatenate, Flatten, Embedding
import numpy as np
import evaluation
import data_management
from keras import initializers


def create_model(num_users, num_items, num_predictive_factors,pretrain):
    #def init_normal( name=None):
    #    return initializers.RandomNormal(mean = 0.0, stddev=0.01, seed=None)
    #initializations.normal(shape, scale=0.01, name=name)

    # embedding size is 2 * num_predictive_factors if MLP is 3 layered

    user_input = Input(shape=(1,), name='user_input')
    item_input = Input(shape=(1,), name='item_input')
    user_embed = Flatten()(Embedding(num_users,
                                     num_predictive_factors * 2,
                                     # W_regularizer = l2(0.01),
                                     input_length=1,
                                     # dropout = 0.3,
                                     name='MLP_user_embed',
                                     embeddings_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01,
                                                                                      seed=None))(user_input))
    item_embed = Flatten()(Embedding(num_items,
                                     num_predictive_factors * 2,
                                     # W_regularizer = l2(0.01),
                                     input_length=1,
                                     # dropout = 0.3,
                                     name='MLP_item_embed',
                                     embeddings_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01,
                                                                                      seed=None))(item_input))
    merged_embed = concatenate([user_embed, item_embed], axis=1)
    mlp_1 = Dense(num_predictive_factors * 4, activation='relu',
                  # W_regularizer = l2(0.01),
                  name='mlp_1')(merged_embed)
    mlp_2 = Dense(num_predictive_factors * 2, activation='relu',
                  # W_regularizer = l2(0.01),
                  name='mlp_2')(mlp_1)
    mlp_3 = Dense(num_predictive_factors, activation='relu',
                  # W_regularizer = l2(0.01),
                  name='mlp_3')(mlp_2)
    main_output = Dense(1,
                        # W_regularizer = l2(0.01),
                        activation='sigmoid', init='lecun_uniform', name='main_output')(mlp_3)
    if pretrain:
        model = Model(inputs=[user_input, item_input], output=main_output)
    else:
        model = Model(inputs=[user_input, item_input], output=mlp_3)
    return model


def train_mlp(num_predictive_factors,batch_size, epochs, interaction_mx, inputs, labels):
    pretrain_model = create_model(num_users=interaction_mx.shape[0],
                                  num_items=interaction_mx.shape[1],
                                  num_predictive_factors=num_predictive_factors,
                                  pretrain=True)
    pretrain_model.compile(optimizer='Adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    pretrain_model.fit(inputs, labels, batch_size, epochs)


    #Save weights for full_model
    wmlp1 = pretrain_model.get_layer('mlp_1').get_weights()
    np.save('MLP_WE/mlp_1_weights_array0', wmlp1[0])
    np.save('MLP_WE/mlp_1_weights_array1', wmlp1[1])
    np.save('MLP_WE/mlp_2_weights', pretrain_model.get_layer('mlp_2').get_weights())
    np.save('MLP_WE/mlp_3_weights', pretrain_model.get_layer('mlp_3').get_weights())
    np.save('MLP_WE/mlp_user_embed_weights', pretrain_model.get_layer('MLP_user_embed').get_weights())
    np.save('MLP_WE/mlp_item_embed_weights', pretrain_model.get_layer('MLP_item_embed').get_weights())

    hit_rate_accuracy = evaluation.evaluate_integer_input('input/testing_data.npy', pretrain_model, 'hit_rate', 'input/int_mat.npy')
    print('MLP produces accuracy rate of: ' + str(hit_rate_accuracy))

if __name__ == '__main__':
    try:
        interaction_mx = np.load('input/int_mat.npy')
    except IOError:
        data_management.load_data()
        interaction_mx = np.load('input/int_mat.npy')
    inputs, labels = data_management.training_data_generation('input/training_data.npy', 'input/int_mat.npy', 5)
    data_management.load_data(file_path='../data/movielens/ratings.dat')
    train_mlp(num_predictive_factors=8, batch_size=256, epochs=2,
              interaction_mx=interaction_mx, inputs=inputs, labels=labels)
