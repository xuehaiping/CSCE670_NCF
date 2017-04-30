import numpy as np
import operator, data_management
from math import log
from operator import div

NDCG_SCORES = np.arange(1,0,-0.1)
def hit_rate(sorted_predictions, target_movie, target_rating):
    movies = [int(i[0]) for i in sorted_predictions]
    #ratings = [int(i[1]) for i in sorted_predictions]
    if target_movie in movies:
        return True



def dcg(predicted_ratings):
    return np.sum(reduce(lambda dcgs, dg: dcgs + [dg+dcgs[-1]],\
                  map(lambda (rank, g): (2**g-1) / log( rank+2, 2), enumerate( predicted_ratings ) ), [0])[1:])

#Make sure predicted_ratings order matches the order for ideal_testing_ratings. In other words, make sure that both
# are ratings for the same movies in the same order but maybe with different values.
def ndcg( predicted_ratings, ideal_testing_ratings ): 
    return dcg(predicted_ratings)/ dcg(ideal_testing_ratings)

def evaluate_rmse(model):
    testing_input, testing_labels = data_management.training_data_generation('input/testing_data.npy', 'input/int_mat.npy', 5)
    #score = model.evaluate(testing_input, testing_labels)
    #score[0]==loss, score[1]==accuracy
    predicted_labels = model.predict(testing_input)
    rmse = np.sqrt(np.mean(np.square(predicted_labels - testing_labels)))
    return rmse

def evaluate_integer_input(fname, model, metric, interactions_matrix):
    target_movies = []
    target_ratings = []
    int_matrix = np.load(interactions_matrix)
    int_matrix = np.delete(int_matrix, 0, 0)
    int_matrix = np.delete(int_matrix,0,1)
    
    lines = np.load(fname)
    
    for line in lines:
        target_movies.append(line[1])
        target_ratings.append(line[2])
    
    summation = 0
    #int_matrix == [[ 0.  0.  1. ...,  0.  0.  4.]]
    
    for idx, user in enumerate(int_matrix):
        #User is a row build from movie's ratings (rather than only 1's)
        
        # pick 100 random non-rated movies 
        zero_indices = np.where(user == 0)[0]
        np.random.shuffle(zero_indices)
        random_indices = zero_indices[0:100] #Movie Ids
        
        # transform into inputs for keras
        movie_vectors = np.append(random_indices, target_movies[idx])
        rating_vectors = np.append(user[random_indices], target_ratings[idx])
        user_vectors = np.repeat([idx + 1], rating_vectors.size, axis=0)
        
        # generate predictions
        predictions = model.predict({'user_input': np.array(user_vectors), 'item_input': np.array(movie_vectors)})
        predictions_idx = dict(zip(movie_vectors, predictions))
        
        #Sorted by ratings
        sorted_predictions = sorted(predictions_idx.items(), key=operator.itemgetter(1), reverse= True)[0:10]
        
        if metric == 'hit_rate':
            if hit_rate(sorted_predictions, target_movies[idx], target_ratings[idx]):
                summation += 1
        elif metric == 'ndcg':
            summation += ndcg(sorted_predictions, target_movies[idx], target_ratings[idx])
        else:
            raise StandardError('metric has to be "hit_rate" or "ndcg"')
    return summation/float(int_matrix.shape[0])
