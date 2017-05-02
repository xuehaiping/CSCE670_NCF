import numpy as np
from keras.models import load_model
import evaluation
# use model.save to save model trained in another script
model = load_model('test.h5')
test = evaluation.evaluate_integer_input('input/testing_data.npy', model, 'ndcg', 'input/int_mat.npy')
rmse1 = evaluation.evaluate_rmse('input/testing_data.npy',model)
print("NDCG: " + str(test))
print("RMSE: " + str(rmse1))
