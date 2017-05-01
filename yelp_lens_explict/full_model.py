import MLP, GMF, data_management, evaluation
from keras.models import Model
from keras.layers import Dense, Embedding, Input, concatenate, multiply, Flatten
import numpy as np
import sys, getopt
import keras

def load_weights(model):
    # changed from model_n to index because model names seem to change.  Look into naming each model.
    model.get_layer(index=5).get_layer('MLP_user_embed').set_weights(np.load('MLP_WE/mlp_user_embed_weights.npy'))
    model.get_layer(index=5).get_layer('MLP_item_embed').set_weights(np.load('MLP_WE/mlp_item_embed_weights.npy'))

    mlp1_0 = np.load('MLP_WE/mlp_1_weights_array0.npy')
    mlp1_1 = np.load('MLP_WE/mlp_1_weights_array1.npy')
    model.get_layer(index=5).get_layer('mlp_1').set_weights([mlp1_0, mlp1_1])

    model.get_layer(index=5).get_layer('mlp_2').set_weights(np.load('MLP_WE/mlp_2_weights.npy'))
    model.get_layer(index=5).get_layer('mlp_3').set_weights(np.load('MLP_WE/mlp_3_weights.npy'))

    model.get_layer(index=6).get_layer('GMF_user_embed').set_weights(np.load('GMF_WE/GMF_user_embed.npy'))
    model.get_layer(index=6).get_layer('GMF_item_embed').set_weights(np.load('GMF_WE/GMF_item_embed.npy'))
    return model

num_predictive_factors = 8
batch_size = 256

num_pretrain_epochs = 2

#p for predcit factors, b for batch size,e for epochs
opts, args = getopt.getopt(sys.argv[1:],"p:b:e:", ["pfactor=","bsize=", "epoch="])
for opt, arg in opts:
    if opt in ("-p", "--pfactor"):
        num_predictive_factors = arg
        print "Number of predictive factors is " + str(num_predictive_factors)
    elif opt in ("-b", "--bsize"):
        batch_size = arg
        print "Batch size is " + str(batch_size)
    elif opt in ("-e", "--epoch"):
        num_pretrain_epochs = arg
        print "number of training epochs for pretrain and full model is " + str(num_pretrain_epochs)

num_final_epochs = num_pretrain_epochs



data_management.load_data()
dimensions = np.load('input/dimensions.npy')
inputs, labels = data_management.training_data_generation('input/training_data.npy', 'input/int_mat.npy', 5)


# pretrain MLP
MLP.train_mlp(num_predictive_factors=num_predictive_factors, batch_size=batch_size, epochs=num_pretrain_epochs,
              dimensions=dimensions, inputs=inputs, labels=labels)
# pretrain GMF
GMF.train_gmf(num_predictive_factors=num_predictive_factors, batch_size=batch_size, epochs=num_pretrain_epochs,
              dimensions=dimensions, inputs=inputs, labels=labels)

# check out the shared vision guide at https://keras.io/getting-started/functional-api-guide/

user_input = Input(shape=(1,), name='user_input')
item_input = Input(shape=(1,), name='item_input')
review_input = Input(shape=(100,), name='review_input')

# ----- MLP Model -----
mlp = MLP.create_model(num_users=dimensions[0],
                       num_items=dimensions[1],
                       num_predictive_factors=num_predictive_factors,
                       pretrain=False)
mlp_output = mlp([user_input, item_input])


# ----- GMF Model -----
gmf = GMF.create_model(num_users=dimensions[0],
                       num_items=dimensions[1],
                       num_predictive_factors=num_predictive_factors,
                       pretrain=False)
gmf_output = gmf([user_input, item_input])

# ----- Paragraph2Vec Model -----
par2vec1 = Dense(num_predictive_factors * 4, activation='relu',
              name='par2vec1')(review_input)
par2vec2 = Dense(num_predictive_factors * 2, activation='relu',
              name='par2vec2')(par2vec1)
par2vec3 = Dense(num_predictive_factors, activation='relu',
              name='par2vec3')(par2vec2)

# ----- Total Model -----
gmf_mlp_par2vec_concatenated = concatenate([mlp_output, gmf_output,par2vec3], axis=1)
NeuMF = Dense(num_predictive_factors * 3, activation='sigmoid', name='NeuMF')(gmf_mlp_par2vec_concatenated)
NeuMF_main_output = Dense(6, activation='softmax', name='NeuMF_main_output')(NeuMF)
model = Model(inputs=[user_input, item_input, review_input], output=NeuMF_main_output)

model = load_weights(model)
model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
labels = keras.utils.to_categorical(labels, 6)
model.fit(inputs, labels, batch_size=batch_size, epochs=num_final_epochs)

model.save('final_model.h5')
