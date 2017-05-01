
# coding: utf-8

# In[18]:

import json


#convert the yelp date data to days as timestamp
def convert_to_days(date):
    item = date.split('-')
    return int(item[0])*365 + int(item[1])*30 + int(item[2]) 





#user counter for generating new user id
user_counter = 1
#restaurant counter for generating new restaurant id
restaurant_counter = 1
users = {}
restaurants = {}
new_yelp_data = []

with open('../data/yelp/training_data.json', 'r') as data_file:
        for line in data_file.readlines():
            yelp_data = []
            data = json.loads(line)
            #preocess user
            if data['user_id'] not in users:
                users[data['user_id']] = user_counter
                yelp_data.append(users[data['user_id']])
                user_counter+=1
            else:
                yelp_data.append(users[data['user_id']])
                
            #process restaurant
            if data['business_id'] not in restaurants:
                restaurants[data['business_id']] = restaurant_counter
                yelp_data.append(restaurants[data['business_id']])
                restaurant_counter+=1
            else:
                yelp_data.append(restaurants[data['business_id']])
                
            #process rating
            yelp_data.append(data['stars'])
            #process time
            yelp_data.append(convert_to_days(data['date']))
            #append review 
            yelp_data.append(data['text'])
            #append the review to the data set
            new_yelp_data.append(yelp_data)

data_file.close()


#write the data into the new file 
with open('../data/yelp/yelp_sample.dat', 'w+') as new_data:
    for line in new_yelp_data:
        new_data.write("%d::%d::%d::%d::%s\n"%(line[0],line[1],line[2],line[3],line[4].encode('ascii', 'ignore').replace('\n', ' ')))

new_data.close()

