import numpy as np
import operator, data_management_yelp
from math import log
from operator import div


def hit_rate(sorted_predictions, target_movie, target_rating):
    movies = [int(i[0]) for i in sorted_predictions]
    #ratings = [int(i[1]) for i in sorted_predictions]
    if target_movie in movies:
        return True

##find the rating from possibile matrix
def find_rating(mx):
    highest = sorted(mx, reverse=True)[0]
    return mx.index(highest)

def dcg(ratings):
    # The first i+1 is because enumerate starts at 0
    dcg_values = [((2 ** v - 1) / log(i + 1 + 1, 2)) for i, v in enumerate(ratings)]
    return np.sum(dcg_values)

def evaluate_rmse(model):
    testing_input, testing_labels = data_management_yelp.training_data_generation('input/testing_data.npy', 'input/int_mat.npy', 5)
    #score = model.evaluate(testing_input, testing_labels)
    #score[0]==loss, score[1]==accuracy
    predicted_labels = model.predict(testing_input)
    rmse = np.sqrt(np.mean(np.square(predicted_labels - testing_labels)))
    return rmse

# Make sure predicted_ratings order matches the order for ideal_testing_ratings. In other words, make sure that both
# are ratings for the same movies in the same order but maybe with different values.
def ndcg(ideal_testing_ratings, predicted_ratings):
    real_ranking = sorted(zip(ideal_testing_ratings, predicted_ratings),reverse=True, key = lambda tp: tp[1])
    real_ranking = [tp[0] for tp in real_ranking]
    return dcg(real_ranking) / dcg(ideal_testing_ratings)

def evaluate_rmse(model):
    testing_input, testing_labels = data_management_yelp.training_data_generation('input/testing_data.npy', 'input/int_mat.npy', 5)
    #score = model.evaluate(testing_input, testing_labels)
    #score[0]==loss, score[1]==accuracy
    predicted_labels = model.predict(testing_input)
    rmse = np.sqrt(np.mean(np.square(predicted_labels - testing_labels)))
    return rmse

def evaluate_integer_input(fname, model, metric, reviews):
    target_movies = {}
    target_ratings = {}
    target_reviews = {}


    lines = np.load(fname)
    reviews = np.load(reviews)
    for index, line in enumerate(lines):
        if line[0] not in target_movies.keys():
            target_movies[line[0]] = [line[1]]
            target_ratings[line[0]] = [line[2]]
            target_reviews[line[0]] = [reviews[index]]


        else:
            target_movies[line[0]].append(line[1])
            target_ratings[line[0]].append(line[2])
            target_reviews[line[0]].append(reviews[index])

    summation = 0
    
    for user, movie in target_movies.iteritems():
        # User is a row build from movie's ratings (rather than only 1's)
        # transform into inputs for keras

        movie_vectors = target_movies[user]
        rating_vectors = target_ratings[user]
        review_vectors = target_reviews[user]
        user_vectors = np.repeat([user], len(rating_vectors), axis=0)

        #movie_vectors_non_sorted = movie_vectors
        #rating_vectors_non_sorte = rating_vectors

        # Sort movies by rating in decreasing order. So movie_vectors, rating_vectors and user_vectors
        movie_rating_user_tuples = [(movie_vectors[i], rating_vectors[i], user_vectors[i], review_vectors[i]) for i, v in
                                    enumerate(movie_vectors)]
        sorted_tuples = sorted(movie_rating_user_tuples, key=lambda item: item[1], reverse=True)

        # Put them back to arrays so we can pass to the NDCG function an ideal ratings vector
        # We do this right before predict, so the predicted labels are returned in this same way, so we only need to pass it to ndcg
        for i, v in enumerate(sorted_tuples):
            movie_vectors[i] = v[0]
            rating_vectors[i] = v[1]
            user_vectors[i] = v[2]
            review_vectors[i] = v[3]
            #print('rating vector length:' + str(len(rating_vectors)))
            #print('rating_vectors[0]: ' + str(rating_vectors[0]))
            #print('review vector length: ' + str(len(review_vectors)))
            #print('review_vectors[0]: ' + str(review_vectors[0]))
            #print('total review_vector: ' + str(review_vectors))

        # generate predictions. This predictions are in the same order as movie_vectors, so we can pass it as it is to the ndcg function
        predictions = model.predict({'user_input': np.array(user_vectors), 'item_input': np.array(movie_vectors), 'review_input': np.array(review_vectors)})
        print(predictions)
        # predictions_idx = dict(zip(movie_vectors, predictions))
        # Sorted by ratings
        # sorted_predictions = sorted(predictions_idx.items(), key=operator.itemgetter(1), reverse= True)[0:5]

        #highest_predictions = []
        #for row in predictions:
        #    highest_predictions.append(find_rating(list(row)))

        if metric == 'hit_rate':
            raise StandardError('Hit rate not suported for rankings"')
            # if hit_rate(sorted_predictions, target_movies[idx], target_ratings[idx]):
            # summation += 1
        elif metric == 'ndcg':
            summation += ndcg(rating_vectors, predictions)
            #if 10< user < 30:
            #    print "rating vectors     " + str(rating_vectors)
            #    #print "predictions" + str(predictions)
            #    print "highest predictions" + str(highest_predictions)
            #    print('--------')
                #print "movie_vectors_non_sorted" + str(movie_vectors_non_sorted)
                #print "rating_vectors_non_sorte" + str(rating_vectors_non_sorte)

        else:
            raise StandardError('metric has to be "ndcg"')
    return summation / float(len(target_movies.keys()))

