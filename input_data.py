import numpy as np

##this block's functions are created for preprocess and training data generation 

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

# In[47]:

##build user dictionary, user list can be created by getting the keys for user dictionary 
user_dict = {}
#test dictionary
test_user = {}
#row number of interaction matrix(user)
row_num = 0
#column number of interaction matrix(movie)
column_num = 0
#movie list
movies = []
#user
users = []
#training data
train_data = []

# read data from ratings.csv, userId, movieId, timestamp

file_path = 'data/movielens/ratings.dat'


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
#assign column number
row_num = max(users) + 1
column_num = max(movies) + 1



#open a training file for writing
with open('input/one_training_data','w+') as tf:
    #append interacted [user,movie,label] pair to the training data
    for usr in user_dict:
        for mov in list(user_dict[usr]):
            tf.write("%s\n" % (str(usr)+','+str(mov)))
tf.close()

#open a training file for writing
with open('input/one_testing_data','w+') as tt:
    #append interacted [user,movie,label] pair to the training data
    for usr in test_user:
        tt.write("%s\n" % (str(usr)+','+str(test_user[usr])))
tt.close()

#interaction matrix
int_mat = interaction_matrix(user_dict,row_num,column_num)
#add test data in the interaction matrix
add_one(test_user,int_mat)
#save the interaction matrix numpy
np.save('input/int_mat', int_mat)
