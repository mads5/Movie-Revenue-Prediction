"""
Created on Mon Mar  5 15:44:10 2018
@author: naveen
"""

# Multiple Linear Regression

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

# all_production_companies will contain all the production_companies with repetition
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

# adding a column genre_score
dataset['genre_score'] = pd.Series(np.random.randn(len(dataset['genres'])), index=dataset.index)

# allocating a particular genre score for a movie
for index,i in zip(dataset.index,dataset['genres']):
    count = 0 
    for j in range(len(i)):
        count += dict_genre[i[j]]
    dataset.loc[index,'genre_score']= count/len(i)
    
    
# adding a column pc_score
dataset['pc_score'] = pd.Series(np.random.randn(len(dataset['genres'])), index=dataset.index)

# allocating a production company score for a movie
for index,i in zip(dataset.index,dataset['production_companies']):
    count = 0 
    for j in range(len(i)):
        count += dict_pc[i[j]]
    dataset.loc[index,'pc_score']= count/len(i)
     
# moving revenue to last
cols = list(dataset.columns.values)
cols.pop(cols.index('revenue')) 
dataset = dataset[cols+['revenue']]

cols = list(dataset.columns.values)
cols.pop(cols.index('original_title')) 
dataset = dataset[['original_title']+cols]


# filling the missing revenue with sum of the avg_genre_scores of that movie
for index,i in zip(dataset.index,dataset['revenue']):
    if(i == 0):
        new_rev = 0
        l = len(dataset['genres'][index])
        for j in dataset['genres'][index]:
            new_rev += dict_genre[j]
        dataset.loc[index,'revenue']= new_rev/l



# deleting unwanted columns
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


X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# calculating rmse
from sklearn.metrics import mean_squared_error
from math import sqrt
rms = sqrt(mean_squared_error(y_test, y_pred))

from sklearn.metrics import explained_variance_score
score = explained_variance_score(y_test, y_pred)










