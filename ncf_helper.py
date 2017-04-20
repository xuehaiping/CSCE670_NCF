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


def hit_rate(sorted_predictions, target):
    movies = [int(i[0]) for i in sorted_predictions]
    if target in movies:
        return True

def NDCG(sorted_predictions, target):
    raise NotImplementedError
    return False

def evaluate(interactions_matrix, model, metric):
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
