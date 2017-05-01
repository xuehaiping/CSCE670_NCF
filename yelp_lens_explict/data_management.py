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
    # write chosen reviews to file
    counter = 0
    with open(output_name, 'w') as output_file:
        with open(input_name, 'rb') as input_file:
            for line in input_file.readlines():
                data = line.split("::", 1)
                if review_counts[data[0]] >= min_reviews_per_user:
                    output_file.write(line)
                    counter += 1
    print ('wrote ' + str(counter) + ' reviews to file')


def training_data_generation(fname, int_mx):
    user_in = []
    movie_in = []
    reviews_in = []
    labels = []
    neg_sample_num = 0
    lines = np.load(fname)
    int_mx = np.load(int_mx)
    # generate postive data
    for data in lines:
        user_in.append(data[0])
        movie_in.append(data[1])
        reviews_in.append(data[2])
        labels.append(int_mx[data[0]][data[1]])

    return {'user_input': np.array(user_in), 'item_input': np.array(movie_in), 'review_input': reviews_in}, np.array(
        labels)


# create index dictionary for an list
def idx_dict(array):
    idx_dict = {}
    for i in range(0, len(array)):
        idx_dict[array[i]] = i
    return idx_dict


# build interaction matrix
def interaction_matrix(u_dict, row, column):
    # u_dict
    # {1: [[914, 3], [1193, 5], [3408, 4]]}
    print('row'+ str(row))
    print('column' + str(column))
    interaction_vector = np.zeros((row, column))
    # create idx dictionary for movie and user
    for usr in u_dict:
        for mov_rating_pair in list(u_dict[usr]):
            interaction_vector[usr][mov_rating_pair[0]] = mov_rating_pair[1]

    return interaction_vector


# add test data to the interaction matrix
def add_one(test_dict, mat):
    for usr in test_dict:
        mat[usr][test_dict[usr]] = 1


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
    row_num = max(users) + 1
    column_num = max(movies) + 1

    # training data
    user_item_triplet = []  # user, movie, rating
    for usr in user_dict:
        for mov in list(user_dict[usr]):
            # mov[0] == id, mov[1]==rating
            user_item_triplet.append([usr, mov[0], mov[1]])

    # testing data
    test_triplet = []
    for usr in test_user:
        for mov in test_user[usr]:
            test_triplet.append([usr, mov[0], mov[1]])

    int_mat = interaction_matrix(user_dict, row_num, column_num)
    # add test data in the interaction matrix
    add_one(test_user, int_mat)
    np.save('input/int_mat', int_mat)
    np.save('input/training_data', user_item_triplet)
    np.save('input/testing_data', test_triplet)


if __name__ == '__main__':
     prune_data('../data/yelp/yelp.dat',
               '../data/yelp/yelp_pruned_20.dat', 20, 0.1)
     #load_data(file_path='../data/yelp/yelp_pruned_20.dat',
     #         review_file_path='input/docvecs.npy')
