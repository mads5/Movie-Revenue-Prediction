#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 19:06:42 2018

@author: Milan, naveen
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import warnings
from scipy.misc import imread
warnings.filterwarnings('ignore')
from IPython.display import HTML
from collections import Counter

# Importing the dataset
dataset=pd.read_csv('tmdb_5000_movies.csv')
mov=pd.read_csv('tmdb_5000_credits.csv')

# changing the genres column from json to string
dataset['genres']=dataset['genres'].apply(json.loads)
for index,i in zip(dataset.index,dataset['genres']):
    list1=[]
    for j in range(len(i)):
        list1.append((i[j]['name']))# the key 'name' contains the name of the genre
    dataset.loc[index,'genres']=str(list1)

dataset['genres']=dataset['genres'].str.strip('[]')
dataset['genres']=dataset['genres'].str.replace(' ','')
dataset['genres']=dataset['genres'].str.replace("'",'')
dataset['genres']=dataset['genres'].str.split(',')

# changing the production company column from json to string
dataset['production_companies']=dataset['production_companies'].apply(json.loads)
for index,i in zip(dataset.index,dataset['production_companies']):
    list1=[]
    for j in range(len(i)):
        list1.append((i[j]['name']))# the key 'name' contains the name of the genre
    dataset.loc[index,'production_companies']=str(list1)

dataset['production_companies']=dataset['production_companies'].str.strip('[]')
dataset['production_companies']=dataset['production_companies'].str.replace(' ','')
dataset['production_companies']=dataset['production_companies'].str.replace("'",'')
dataset['production_companies']=dataset['production_companies'].str.split(',')

# all_genres will contain all the genres with repetition
# unique_genres will contain unique_genres
unique_genres = []
all_genres = []

for index,i in zip(dataset.index,dataset['genres']):
    for j in range(len(i)):
        if i[j] not in unique_genres:
            unique_genres.append(i[j])
        all_genres.append(i[j])

# creating genre columns
        
dataset = dataset.join(pd.DataFrame(
    {
        'Action': 0,
        'Action': 0,
        'Adventure': 0,
        'Fantasy': 0,
        'ScienceFiction': 0,
        'Crime': 0,
        'Drama': 0,
        'Thriller': 0,
        'Animation': 0,
        'Family': 0,
        'Western': 0,
        'Comedy': 0,
        'Romance': 0,
        'Horror': 0,
        'Mystery': 0,
        'History': 0,
        'War': 0,
        'Music': 0,
        'Documentary': 0,
        'Foreign': 0,
        'TVMovie': 0
    }, index=dataset.index
))

# filling the genres column

for index,i in zip(dataset.index,dataset['genres']):
    for j in range(len(i)):
        list0 = i[j] 
        dataset.loc[index,list0] = 1

# all_production_companies will contain all the production_companies
# with repetition
# unique_production_companies will contain unique
unique_pc = []
all_pc = []

for index,i in zip(dataset.index,dataset['production_companies']):
    for j in range(len(i)):
        if i[j] not in unique_pc:
            unique_pc.append(i[j])
        all_pc.append(i[j])


# allocating score for the genres
dict_genre = {}
count_genre = {}

for j in range(len(unique_genres)):
    dict_genre[unique_genres[j]] = 0;
    count_genre[unique_genres[j]] = 0;
for index,i in zip(dataset.index,dataset['genres']):
    for j in range(len(i)):
        dict_genre[i[j]] += dataset['revenue'][index]
        count_genre[i[j]] = count_genre[i[j]] + 1

for j in range(len(unique_genres)):
    dict_genre[unique_genres[j]] = dict_genre[unique_genres[j]]/count_genre[unique_genres[j]]; 

# allocating score for the production_companies
dict_pc = {}
count_pc = {}

for j in range(len(unique_pc)):
    dict_pc[unique_pc[j]] = 0;
    count_pc[unique_pc[j]] = 0;
for index,i in zip(dataset.index,dataset['production_companies']):
    for j in range(len(i)):
        dict_pc[i[j]] += dataset['revenue'][index]
        count_pc[i[j]] = count_pc[i[j]] + 1

for j in range(len(unique_pc)):
    dict_pc[unique_pc[j]] = dict_pc[unique_pc[j]]/count_pc[unique_pc[j]]; 

# filling the missing revenue with sum of the avg_genre_scores of that movie
"""for index,i in zip(dataset.index,dataset['revenue']):
    if(i == 0):
        new_rev = 0
        l = len(dataset['genres'][index])
        for j in dataset['genres'][index]:
            new_rev += dict_genre[j]
        dataset.loc[index,'revenue']= new_rev/l
"""
# Faulty Row
dataset.loc[4553, 'release_date'] = '2000-01-1'

# Extracting date of release
release_date_1 = []
for i1 in range(0,len(dataset)):
    i = dataset.loc[i1,'release_date']
    a = 1000*int(i[0]) + 100*(int(i[1])) + 10*(int(i[2])) + int(i[3])
    if(a<1956):
        a = 1956    
    release_date_1.append(a)

# Importing inflation data and making revenue to a base price year 1956
inflation_stuff = pd.read_csv("year_and_inflation.csv")
n1 = len(inflation_stuff)
inflation_stuff.loc[n1-1, 'inflation'] = 1
for i in range(1,n1):
    inflation_stuff.loc[n1 - 1 - i, 'inflation'] =  inflation_stuff.loc[n1-i,'inflation']*(1 + (inflation_stuff.loc[n1 - i - 1, 'inflation']/100))
    
for i in range(0,len(dataset)):
    dataset.loc[i,'revenue'] /= inflation_stuff.loc[2017 - release_date_1[i], 'inflation']
    
# Extracting Month
month_release = []
for i1 in range(0,len(dataset)):
    i = dataset.loc[i1,'release_date']
    a = 10*(int(i[5])) + int(i[6])
    month_release.append(a)


# divide the renevue into 10 groups
rev = np.array(dataset['revenue']) 
s = len(dataset['revenue'])
p1 = s/10;
p2 = 2*(s/10);
p3 = 3*(s/10);
p4 = 4*(s/10);
p5 = 5*(s/10);
p6 = 6*(s/10);
p7 = 7*(s/10);
p8 = 8*(s/10);
p9 = 9*(s/10);
p10 = 10*(s/10);

p1 = int(round(p1))
p2 = int(round(p2))
p3 = int(round(p3))
p4 = int(round(p4))
p5 = int(round(p5))
p6 = int(round(p6))
p7 = int(round(p7))
p8 = int(round(p8))
p9 = int(round(p9))
p10 = int(round(p10))

rev = np.sort(rev)
r1 = 2000000
r2 = 10000000
r3 = 20000000
r4 = 40000000
r5 = 65000000
r6 = 100000000
r7 = 150000000
r8 = 200000000



# creating a column for revenue class
dataset['revenue_class'] = pd.Series(np.random.randn(len(dataset['revenue'])), index=dataset.index)

# putting the class values got from the range
for index,i in zip(dataset.index,dataset['revenue_class']):
    c = 0;
    val = dataset['revenue'][index];
    if (val <= r1):
        c = 1;
    elif (val > r1 and val <= r2):
        c = 2;
    elif (val > r2 and val <= r3):
        c = 3;
    elif (val > r3 and val <= r4):
        c = 4;
    elif (val > r4 and val <= r5):
        c = 5;
    elif (val > r5 and val <= r6):
        c = 6;
    elif (val > r6 and val <= r7):
        c = 7;
    elif (val > r7 and val <= r8):
        c = 8;
    else:
        c = 9;
    
    dataset.loc[index,'revenue_class']= c

del dataset['homepage']
del dataset['id']
del dataset['keywords']
del dataset['original_language']
del dataset['overview']
del dataset['production_countries']
del dataset['runtime']
del dataset['spoken_languages']
del dataset['status']
del dataset['tagline']
del dataset['title']
del dataset['genres']
del dataset['production_companies']
del dataset['release_date']
del dataset['revenue']

dataset.drop(dataset.columns[25], axis=1, inplace = True)

# Adding month of the release
dataset = dataset.join(pd.DataFrame({'Month': 1},index=dataset.index))
for i in range(0,len(dataset)):
    dataset.loc[i,'Month'] = month_release[i]

cols = list(dataset.columns.values)
cols.pop(cols.index('original_title')) 
dataset = dataset[['original_title']+cols]

X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Encoding the Dependent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
y_train = y_train.reshape(-1, 1)
labelencoder_y_train = LabelEncoder()
y_train[:, 0] = labelencoder_y_train.fit_transform(y_train[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
y_train = onehotencoder.fit_transform(y_train).toarray()

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
y_test = y_test.reshape(-1, 1)
labelencoder_y_test = LabelEncoder()
y_test[:, 0] = labelencoder_y_test.fit_transform(y_test[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
y_test = onehotencoder.fit_transform(y_test).toarray()


# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 23)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 10, init = 'uniform', activation = 'relu', input_dim = 23))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 10, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 10, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 9, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Convert labels to categorical one-hot encoding
# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 30, nb_epoch = 100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

y_pred_max_index = np.argmax(y_pred, axis = 1)
y_test_max_index = np.argmax(y_test, axis = 1)
# testing training set accuracy
y_pred_2 = classifier.predict(X_train)
y_pred_2_max_index = np.argmax(y_pred_2, axis = 1)
y_train_max_index = np.argmax(y_train, axis = 1)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test_max_index, y_pred_max_index)

from sklearn.metrics import accuracy_score
test_ac = accuracy_score(y_test_max_index, y_pred_max_index) 

from sklearn.metrics import accuracy_score
train_ac = accuracy_score(y_train_max_index, y_pred_2_max_index)
