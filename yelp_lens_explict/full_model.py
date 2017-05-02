import MLP, GMF, data_management_yelp, evaluation_yelp, doc2vec
from keras.models import Model
from keras.layers import Dense, Embedding, Input, concatenate, multiply, Flatten
import numpy as np
import sys, getopt, os.path
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
        num_predictive_factors = int(arg)
        print "Number of predictive factors is " + str(num_predictive_factors)
    elif opt in ("-b", "--bsize"):
        batch_size = int(arg)
        print "Batch size is " + str(batch_size)
    elif opt in ("-e", "--epoch"):
        num_pretrain_epochs = int(arg)
        print "number of training epochs for pretrain and full model is " + str(num_pretrain_epochs)

num_final_epochs = num_pretrain_epochs
formatted_yelp_data = '../data/yelp/yelp.dat'
pruned_yelp_data = '../data/yelp/yelp_pruned_20.dat'
all_docvecs = 'input/docvecs.npy'
dimensions_file = 'input/dimensions.npy'
# TODO: improve this
if not os.path.isfile(pruned_yelp_data):
    data_management_yelp.prune_data(formatted_yelp_data,pruned_yelp_data, 20, 0.5)
    doc2vec.doc2vec(pruned_yelp_data, all_docvecs)
if not os.path.isfile(all_docvecs):
    doc2vec.doc2vec(pruned_yelp_data, all_docvecs)
    data_management_yelp.load_data(file_path=pruned_yelp_data, review_file_path=all_docvecs)
if not os.path.isfile(dimensions_file):
    data_management_yelp.load_data(file_path=pruned_yelp_data, review_file_path=all_docvecs)
dimensions = np.load(dimensions_file)
inputs, labels = data_management_yelp.training_data_generation('input/training_data.npy', 'input/training_reviews.npy')


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
NeuMF_main_output = Dense(1, activation='relu', name='NeuMF_main_output')(NeuMF)
model = Model(inputs=[user_input, item_input, review_input], output=NeuMF_main_output)

model = load_weights(model)
model.compile(optimizer='sgd',
              loss='mean_squared_error',
              metrics=['accuracy'])
model.fit(inputs, labels, batch_size=batch_size, epochs=num_final_epochs)

ndcg = evaluation_yelp.evaluate_integer_input('input/testing_data.npy', model, 'ndcg', 'input/testing_reviews.npy')
rmse = evaluation_yelp.evaluate_rmse('input/testing_data.npy','input/testing_reviews.npy',model)
print("NDCG: " + str(ndcg))
print("RMSE: " + str(rmse))

file_name = 'output/movie_lens_' + 'p-' + str(num_predictive_factors) + 'b-' + str(batch_size) + 'e-' + str(num_pretrain_epochs)
with open(file_name,'w+') as ofile:
    #hit = "hit rate: " + str(hit_rate_accuracy) +'\n'
    #ofile.write(hit)
    n = "NDCG: " + str(ndcg) + '\n'
    ofile.write(n)
print(ndcg)
model_name = 'output_model/movie_lens_' + 'p-' + str(num_predictive_factors) + 'b-' + str(batch_size) + 'e-' + str(num_pretrain_epochs) + '.h5'
model.save(model_name)
