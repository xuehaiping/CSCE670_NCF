import numpy as np
from keras.models import load_model
import ncf_helper as helper

# use model.save to save model trained in another script

model = load_model('MLP_for_eval.h5')
interaction_mx = np.load('interaction_mx.npy')
test = helper.evaluate(interaction_mx, model, 'hit_rate')
print('accuracy rate of: ' + str(test))