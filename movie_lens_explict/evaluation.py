import numpy as np
import operator


def hit_rate(sorted_predictions, target):
    movies = [int(i[0]) for i in sorted_predictions]
    if target in movies:
        return True


def ndcg(sorted_predictions, target):
    raise NotImplementedError
    return False


def evaluate_integer_input(fname, model, metric, interactions_matrix):
    target_movies = []
    int_matrix = np.load(interactions_matrix)
    int_matrix = np.delete(int_matrix, 0, 0)
    int_matrix = np.delete(int_matrix,0,1)
    lines = np.load(fname)
    for line in lines:
        target_movies.append(line[1])
    summation = 0
    for idx, user in enumerate(int_matrix):
        # pick 100 random non-interacted movies
        zero_indices = np.where(user == 0)[0]
        np.random.shuffle(zero_indices)
        random_indices = zero_indices[0:100]
        # transform into inputs for keras
        movie_vectors = np.append(random_indices, target_movies[idx])
        user_vectors = np.repeat([idx + 1], movie_vectors.size, axis=0)
        # generate predictions
        predictions = model.predict({'user_input': np.array(user_vectors), 'item_input': np.array(movie_vectors)})
        predictions_idx = dict(zip(movie_vectors, predictions))
        sorted_predictions = sorted(predictions_idx.items(), key=operator.itemgetter(1), reverse= True)[0:10]
        if metric == 'hit_rate':
            if hit_rate(sorted_predictions, target_movies[idx]):
                summation += 1
        # TODO: implement NDCG
        elif metric == 'ndcg':
            summation += 1
        else:
            raise StandardError('metric has to be "hit_rate" or "ndcg"')
    return summation/float(int_matrix.shape[0])
