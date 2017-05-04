import numpy as np
from keras.models import load_model
import evaluation_yelp as helper

# use model.save to save model trained in another script

model = load_model('final_model.h5')
test = helper.evaluate_integer_input('input/testing_data.npy', model, 'ndcg', 'input/testing_reviews.npy')
print test
