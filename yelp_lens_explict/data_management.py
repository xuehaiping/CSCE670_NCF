import numpy as np
import random


##this block's functions are created for preprocess and training data generation


def prune_data(input_name, output_name, min_reviews_per_user, percent):
    """

    :param input_name:
    :param output_name:
    :param min_reviews_per_user:
    :param percent: what % of the eligible data to keep
    :return:
    """
    # count how many reviews/user
    random.seed(3)
    review_counts = dict()
    users = {}
    restaurants = {}
    with open(input_name, 'rb') as input_file:
        for i in input_file.readlines():
            data = i.split("::", 1)
            if data[0] in review_counts:
                review_counts[data[0]] += 1
            else:
                review_counts[data[0]] = 1
    # randomly select according to percent param
    for key, value in review_counts.iteritems():
        if value >= min_reviews_per_user and random.random() > percent:
            review_counts[key] = -1
    # reassign user and item IDs
    user_counter = 0
    restaurant_counter = 0
    written_counter = 0
    with open(output_name, 'w') as output_file:
        with open(input_name, 'rb') as input_file:
            for i in input_file.readlines():
                data = i.split("::", 2)
                if review_counts[data[0]] >= min_reviews_per_user:
                    line_data = []
                    if data[0] not in users:
                        users[data[0]] = user_counter
                        line_data.append(users[data[0]])
                        user_counter += 1
                    else:
                        line_data.append(users[data[0]])

                    # process restaurant
                    if data[1] not in restaurants:
                        restaurants[data[1]] = restaurant_counter
                        line_data.append(restaurants[data[1]])
                        restaurant_counter += 1
                    else:
                        line_data.append(restaurants[data[1]])
                    # process everything else
                    line_data.append(data[2])

                    output_file.write("%d::%d::%s" % (line_data[0], line_data[1], line_data[2]))
                    written_counter += 1

    print ('wrote ' + str(written_counter) + ' reviews to file')


def training_data_generation(fname,reviews_input):
    user_in = []
    movie_in = []
    #reviews_in = []
    labels = []
    lines = np.load(fname)
    reviews = np.load(reviews_input)
    # generate postive data
    for data in lines:
        user_in.append(data[0])
        movie_in.append(data[1])
        #reviews_in.append(data[2])
        labels.append(data[2])

    return {'user_input': np.array(user_in), 'item_input': np.array(movie_in), 'review_input': reviews}, np.array(
        labels)


def load_data(file_path, review_file_path):
    # build user dictionary, user list can be created by getting the keys for user dictionary
    user_dict = {}
    # test dictionary
    test_user = {}
    movies = []
    reviews = np.load(review_file_path)

    # read data from ratings.csv, userId, movieId, timestamp
    # UserID::MovieID::Rating::Timestamp
    ## create user dictionary
    with open(file_path, 'rb') as f:
        index = 0
        for i in f.readlines():
            data = i.split("::")
        # data[0]==user, data[1]==movie, data[2]==rating, data[3]==timestamp
        #for i in range(1, 20000):
        #    data = f.readline().split("::")

            # add movie to movie list
            if int(data[1]) not in movies:
                movies.append(int(data[1]))
            # add user data into dictionary (movieId, rating)
            if int(data[0]) in user_dict:
                user_dict[int(data[0])].append((int(data[1]), int(data[2]), int(data[3]), reviews[index]))
            else:
                user_dict[int(data[0])] = list()  # First user appearence
                user_dict[int(data[0])].append((int(data[1]), int(data[2]), int(data[3]), reviews[index]))

            index += 1
            if index % 1000 == 0:
                print(index)
    f.close()
    print('read into user_dict')

    # pick out the lastest movie the user watch and add it to test dictionary
    for index,user in enumerate(user_dict):
        # movie[2] == timestamp
        movie_list = sorted(user_dict[user], key=lambda movie: movie[2], reverse=True)
        test_user[user] = []
        # pull five movie out
        for i in range(0, 5):
            test_user[user].append([movie_list[0][0], movie_list[0][1], movie_list[0][3]])
            movie_list.pop(0)
        # add the training data to dictionary
        movie_rating_list = [[movie[0], movie[1], movie[3]] for movie in movie_list]
        user_dict[user] = movie_rating_list
    print('split into test/train')
    # convert it back to list
    movies = list(movies)
    users = user_dict.keys()
    # assign row and column numbers
    row_num = len(users)
    column_num = len(movies)

    # training data
    user_item_triplet = []  # user, movie, rating
    for usr in user_dict:
        for mov in list(user_dict[usr]):
            # mov[0] == id, mov[1]==rating
            user_item_triplet.append([usr, mov[0], mov[1]])

    # training_reviews
    training_reviews = []  # user, movie, rating
    for usr in user_dict:
        for mov in list(user_dict[usr]):
            # mov[0] == id, mov[1]==rating
            training_reviews.append(mov[2])

    # testing data
    test_triplet = []
    for usr in test_user:
        for mov in test_user[usr]:
            test_triplet.append([usr, mov[0], mov[1]])

    # testing_reviews
    testing_reviews = []  # user, movie, rating
    for usr in user_dict:
        for mov in list(user_dict[usr]):
            # mov[0] == id, mov[1]==rating
            testing_reviews.append(mov[2])

    np.save('input/training_data', user_item_triplet)
    np.save('input/training_reviews', training_reviews)
    np.save('input/testing_data', test_triplet)
    np.save('input/testing_reviews', testing_reviews)
    np.save('input/dimensions', np.array([row_num, column_num]))


if __name__ == '__main__':
     #prune_data('../data/yelp/yelp.dat',
     #         '../data/yelp/yelp_pruned_20.dat', 20, 0.1)
     load_data(file_path='../data/yelp/yelp_pruned_20.dat',
              review_file_path='input/docvecs.npy')
