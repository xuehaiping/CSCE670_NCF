import numpy as np
import operator

def preprocess_data(users_matrix, items_matrix, interactions_matrix, batch_size):
    if (interactions_matrix.size - len(users_matrix[0])) % batch_size != 0:
        raise StandardError(str(interactions_matrix.size - len(users_matrix[0])) + 'is not divisible by ' + str(batch_size))
    users = []
    items = []
    interactions = []
    while True:
        for user_idx, user in enumerate(users_matrix):
            for item_idx, item in enumerate(items_matrix):
                if interactions_matrix[user_idx][item_idx] >= 0:
                    users.append(user)
                    items.append(item)
                    interactions.append(interactions_matrix[user_idx][item_idx])
                    if len(users) == batch_size:
                        yield ({'user_input': np.array(users), 'item_input': np.array(items)},
                               np.array(interactions))
                        users = []
                        items = []
                        interactions = []


def generate_one_hot(id, total):
    vector = np.zeros(total)
    vector[id] = 1.0
    return vector

#generate training data 
def training_data_generation(fname,int_mx, times):
    user_in = []
    movie_in = []
    labels = []
    neg_sample_num = 0
    lines = np.load(fname)
    #generate postive data 
    neg_sample_num = len(lines) * times 
    for data in lines:
        user_in.append(data[0])
        movie_in.append(data[1])
        labels.append(1)
    #generate random samples
    row, column = np.where(int_mx == 0)
    indices = list(zip(row, column))
    np.random.shuffle(indices)
    random_indices = indices[0:neg_sample_num]
    
    for data in random_indices:
        user_in.append(data[0])
        movie_in.append(data[1])
        labels.append(0)         
            
    return {'user_input': np.array(user_in), 'item_input': np.array(movie_in)},np.array(labels) 

def hit_rate(sorted_predictions, target):
    movies = [int(i[0]) for i in sorted_predictions]
    if target in movies:
        return True

def NDCG(sorted_predictions, target):
    raise NotImplementedError
    return False

def evaluate(fname, model, metric,interactions_matrix = None):
    summation = 0
    for idx, user in enumerate(interactions_matrix):
        # pick 100 random non-interacted movies
        zero_indices = np.where(user == 0)[0]
        np.random.shuffle(zero_indices)
        random_indices = zero_indices[0:100]
        latest_movie = np.where(user < 0)[0]
        all_indices = np.append(random_indices, latest_movie)
        # re-generate one-hot representations
        user_one_hot = generate_one_hot(idx, interactions_matrix.shape[0])
        user_vectors = np.repeat([user_one_hot], all_indices.size, axis=0)
        movie_vectors = []
        for movie in all_indices:
            movie_vectors.append(generate_one_hot(movie, interactions_matrix.shape[1]))

        # generate predictions
        predictions = model.predict({'user_input': np.array(user_vectors), 'item_input': np.array(movie_vectors)})
        # TODO: make sure axis is correct
        predictions_idx = dict(zip(all_indices, predictions))
        sorted_predictions = sorted(predictions_idx.items(), key=operator.itemgetter(1))[0:10]
        if metric == 'hit_rate':
            if hit_rate(sorted_predictions, idx):
                summation += 1
        # TODO: implement NDCG
        elif metric == 'ndcg':
            summation += 1
        else:
            raise StandardError('metric has to be "hit_rate" or "ndcg"')
    return summation / float(interactions_matrix.shape[0])


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
        movie_vectors = np.append(random_indices, target_movies[idx])
        #movie_vectors = np.repeat(target_movies[idx], 101, axis=0)
        # re-generate one-hot representations
        user_vectors = np.repeat([idx + 1], movie_vectors.size, axis=0)
        # generate predictions
        predictions = model.predict({'user_input': np.array(user_vectors), 'item_input': np.array(movie_vectors)})
        #print('user: ' + str(idx + 1) + ':' + str(predictions[0]))
        # TODO: make sure axis is correct
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
