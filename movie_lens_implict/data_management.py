import numpy as np

##this block's functions are created for preprocess and training data generation 

def training_data_generation(fname, int_mx, times):
    user_in = []
    movie_in = []
    labels = []
    neg_sample_num = 0
    lines = np.load(fname)
    int_mx = np.load(int_mx)
    # generate postive data
    neg_sample_num = len(lines) * times
    for data in lines:
        user_in.append(data[0])
        movie_in.append(data[1])
        labels.append(1)
    # generate random samples
    row, column = np.where(int_mx == 0)
    indices = list(zip(row, column))
    np.random.shuffle(indices)
    random_indices = indices[0:neg_sample_num]
    for data in random_indices:
        user_in.append(data[0])
        movie_in.append(data[1])
        labels.append(0)

    return {'user_input': np.array(user_in), 'item_input': np.array(movie_in)}, np.array(labels)

#create index dictionary for an list
def idx_dict(array):
    idx_dict = {}
    for i in range(0,len(array)):
        idx_dict[array[i]] = i
    return idx_dict

                    
##add -1 to indicate the (user, movie) pair are used for testing purpose                    
def add_neg_one(test_dict, int_mx, movs, usrs):
    movie_idx_dict = idx_dict(movs)
    user_idx_dict = idx_dict(usrs)
    for usr in test_dict:
        if usr in user_idx_dict:
            int_mx[user_idx_dict[usr]][movie_idx_dict[test_dict[usr]]] = -1

            
#build interaction matrix
def interaction_matrix(u_dict,row,column): 
    interaction_vector = np.zeros((row, column))
    #create idx dictionary for movie and user
    for usr in u_dict:
        for mov in list(u_dict[usr]):
             interaction_vector[usr][mov] = 1
    return interaction_vector

#add test data to the interaction matrix 
def add_one(test_dict, mat):
    for usr in test_dict:
        mat[usr][test_dict[usr]] = 1


def load_data(file_path='../data/ratings.dat'):
    #build user dictionary, user list can be created by getting the keys for user dictionary
    user_dict = {}
    #test dictionary
    test_user = {}
    movies = []

    # read data from ratings.csv, userId, movieId, timestamp

    ##create user dictionary
    with open(file_path, 'rb') as f:
        for i in f.readlines():
            data = i.split("::")
        #for i in range(1,200):
            #data = f.readline().split("::")

            #add movie to movie list
            if int(data[1]) not in movies:
                movies.append(int(data[1]))
            #add data into dictionary
            if int(data[0]) in user_dict:
                user_dict[int(data[0])].append( (int(data[1]), int(data[3])))
            else:
                user_dict[int(data[0])] = list()
                user_dict[int(data[0])].append((int(data[1]), int(data[3])))
    f.close()


    # pick out the lastest movie the user watch and add it to test dictionary
    for user in user_dict:
        movie_list = sorted(user_dict[user], key=lambda movie: movie[1], reverse=True)
        test_user[user] = movie_list[0][0]
        movie_list.pop(0)
        movie_list = [movie[0] for movie in movie_list]
        user_dict[user] = set(movie_list)

    # convert it back to list
    movies = list(movies)
    users = user_dict.keys()
    # assign row and column numbers
    row_num = max(users) + 1
    column_num = max(movies) + 1

    # training data
    user_item_pair = []
    for usr in user_dict:
        for mov in list(user_dict[usr]):
            user_item_pair.append([usr, mov])

    # testing data
    test_pair = []
    for usr in test_user:
        test_pair.append([usr,test_user[usr]])

    int_mat = interaction_matrix(user_dict,row_num,column_num)
    # add test data in the interaction matrix
    add_one(test_user,int_mat)
    np.save('input/int_mat', int_mat)
    np.save('input/training_data', user_item_pair)
    np.save('input/testing_data', test_pair)
