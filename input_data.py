import numpy as np

#create index dictionary for an list
def idx_dict(array):
    idx_dict = {}
    for i in range(0,len(array)):
        idx_dict[array[i]] = i
    return idx_dict

#build interaction matrix
def interaction_matrix(u_dict, movs, usrs): 
    interaction_vector = np.zeros((len(usrs), len(movs)))
    #create idx dictionary for movie and user
    movie_idx_dict = idx_dict(movs)
    user_idx_dict = idx_dict(usrs)
    for usr in usrs:
        for mov in list(u_dict[usr]):
             interaction_vector[user_idx_dict[usr]][movie_idx_dict[mov]] = 1
    return interaction_vector

#create one hot vector for a list 
def one_hot_vector(array):
    one_vector = np.zeros((len(array), len(array)))
    for i in range(len(array)):
        one_vector[i][i] = 1
    return one_vector

##add -1 to indicate the (user, movie) pair are used for testing purpose                    
def add_neg_one(test_dict, int_mx, movs, usrs):
    movie_idx_dict = idx_dict(movs)
    user_idx_dict = idx_dict(usrs)
    for usr in test_dict:
        if usr in user_idx_dict:
            int_mx[user_idx_dict[usr]][movie_idx_dict[test_dict[usr]]] = -1
        

# In[47]:

##build user dictionary, user list can be created by getting the keys for user dictionary 
user_dict = {}
#set for movies
movies = set()
#user_list
users = []
#test dictionary
test_user = {}

# read data from ratings.csv, userId, movieId, timestamp
file_path = 'ratings.csv'

##create user dictionary
with open(file_path, 'rb') as f:
    f.readline()
    for i in range(1,200):
        data = f.readline().split(",")
        #add movie to movie list
        if int(data[1]) not in movies:
            movies.add(int(data[1]))   
        #add data into dictionary
        if int(data[0]) in user_dict:
            user_dict[int(data[0])].append( (int(data[1]), int(data[3])))
        else:
            user_dict[int(data[0])] = list()
            user_dict[int(data[0])].append((int(data[1]), int(data[3])))
f.close()


#pick out the lastest movie the user watch and add it to test dictionary
for user in user_dict:
    movie_list = sorted(user_dict[user], key=lambda movie: movie[1], reverse=True)
    test_user[user] = movie_list[0][0]
    movie_list.pop(0)
    movie_list = [movie[0] for movie in movie_list]
    user_dict[user] = set(movie_list)
    

#convert it back to list
movies = list(movies)
#add users Id from use dictionary 
users = user_dict.keys()

#interaction matrix 
interact_mx = interaction_matrix(user_dict, movies, users)
#create one hot matrix for user
one_hot_user = one_hot_vector(users)
#create one hot matrix for movies
one_hot_movies = one_hot_vector(movies)
#add negative one to the interaction matrix
add_neg_one(test_user, interact_mx, movies, users)

np.save('interaction_mx', interact_mx)
np.save('one_hot_user', one_hot_user)
np.save('one_hot_movies', one_hot_movies)

