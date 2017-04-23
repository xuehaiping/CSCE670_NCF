import numpy as np
from keras.models import load_model
import ncf_helper as helper

# use model.save to save model trained in another script

model = load_model('MLP.h5')
test = helper.evaluate_integer_input('input/testing_data.npy', model, 'hit_rate', 'input/int_mat.npy')
print('accuracy rate of: ' + str(test))