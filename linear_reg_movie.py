# -*- coding: utf-8 -*-
#author: Milan Jolly

#------------------------Prepocessing Steps-----------------------------------#

# Importing the Important Libraries
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')
import base64
import io
from scipy.misc import imread
import codecs
from IPython.display import HTML
from collections import Counter

# For converting the string representation of 
# dict to dict
import ast

# Extracting the data from csv files
movies=pd.read_csv('tmdb_5000_movies.csv')
mov=pd.read_csv('tmdb_5000_credits.csv')
mo = movies

# changing the genres column from json to string
movies['genres']=movies['genres'].apply(json.loads)
for index,i in zip(movies.index,movies['genres']):
    list1=[]
    for j in range(len(i)):
        list1.append((i[j]['name']))# the key 'name' contains the name of the genre
    movies.loc[index,'genres']=str(list1)

movies['genres']=movies['genres'].str.strip('[]')
movies['genres']=movies['genres'].str.replace(' ','')
movies['genres']=movies['genres'].str.replace("'",'')
movies['genres']=movies['genres'].str.split(',')

list2 = []

# The loop to get the maximum number of distinct genres
for index,i in zip(movies.index,movies['genres']):
    for j in range(len(i)):
        if i[j] not in list2:
            list2.append(i[j])
            
no_of_genres = len(list2)

# Forming the genre_binary_list
movies['genre_binary_list'] = movies.index

for index,i in zip(movies.index,movies['genres']):
    samlist = np.zeros(21,dtype=int);
    for j in range(len(list2)):
        if list2[j] in movies.loc[index,'genres']:
            samlist[j] = 1
    movies.loc[index,'genre_binary_list' ] = str(samlist)

movies['genre_binary_list']=movies['genre_binary_list'].str.strip('[]')
movies['genre_binary_list']=movies['genre_binary_list'].str.split(' ')

m = []

# Converting the dict-like-string to dict
for j in range(0,len(mov['crew'])):
    m.append( ast.literal_eval( mov['crew'][j]) )

mov.crew = m

m = []

# Converting the dict-like-string to dict
for j in range(0,len(mov['cast'])):
    m.append( ast.literal_eval( mov['cast'][j]) )

mov.cast = m

dataset = movies
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
    
for i in range(0,len(dataset)):
    dataset.loc[i,'budget'] /= inflation_stuff.loc[2017 - release_date_1[i], 'inflation']
    
# Extracting Month
month_release = []
for i1 in range(0,len(dataset)):
    i = dataset.loc[i1,'release_date']
    a = 10*(int(i[5])) + int(i[6])
    month_release.append(a)
    
dataset = dataset.join(pd.DataFrame(
    {
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
                   
# Filling the cast and crew scores down below:                   
director_list = []
writer_list = []
editor_list = []
producer_list = []

for i,i1 in zip(mov.crew, mov.index):
    dire = 1
    writ = 1
    edit = 1
    prod = 1
    director_list.append('Someone')
    writer_list.append('Someone')
    editor_list.append('Someone')
    producer_list.append('Someone')
    
    for j in i:
        if (j['job'] == 'Director' and dire):
            if 'name' in list(j.keys()):
              director_list[i1] = j['name']
              dire = 0
        
        elif (j['job'] == 'Editor' and edit):
          if 'name' in list(j.keys()):  
            editor_list[i1] = j['name'] 
            edit = 0
            
        elif (j['job'] == 'Writer' and writ):
          if 'name' in list(j.keys()):
            writer_list[i1] = j['name']
            writ = 0
            
        elif (j['job'] == 'Producer' and prod):
          if 'name' in list(j.keys()):  
            producer_list[i1] = j['name']
            prod = 0

director_scores1 = scores(director_list, list(dataset.revenue), list(dataset.vote_average))
editor_scores = scores(editor_list, list(dataset.revenue), list(dataset.vote_average))
writer_scores = scores(writer_list, list(dataset.revenue), list(dataset.vote_average))
dataset = dataset.join(pd.DataFrame({
        'Director_score': 0,
        'Editor_score': 0,
        'Writer_score': 0,
        'Cast1': 0,
        'Cast2': 0,
        'Cast3': 0,
        'Cast4': 0,
        'Cast5': 0
        }, index = mov.index))
    
cast1 = []
cast2 = []
cast3 = []
cast4 = []
cast5 = []

for i in range(0,len(mov)): 
    cast1.append('Someone')
    cast2.append('Someone')
    cast3.append('Someone')
    cast4.append('Someone')
    cast5.append('Someone')

for i in range(0,4803):    
    if i == 4553:
        continue
    n = len(mov['cast'][i])    
    if n>=1:
        cast1[i] = mov['cast'][i][0]['name']    
    if n>=2:
        cast2[i] = mov['cast'][i][1]['name']    
    if n>=3:
        cast3[i] = mov['cast'][i][2]['name']    
    if n>=4:
        cast4[i] = mov['cast'][i][3]['name']    
    if n>=5:
        cast5[i] = mov['cast'][i][4]['name']
        
cast1_score = scores(cast1, list(dataset.revenue), list(dataset.popularity/10))                                              
cast2_score = scores(cast2, list(dataset.revenue), list(dataset.popularity/10))
cast3_score = scores(cast3, list(dataset.revenue), list(dataset.popularity/10))
cast4_score = scores(cast4, list(dataset.revenue), list(dataset.popularity/10))
cast5_score = scores(cast5, list(dataset.revenue), list(dataset.popularity/10))

for i in range(0,len(mov)):
    if director_list[i] in list(director_scores1.keys()):
      dataset.loc[i, 'Director_score'] = director_scores1[director_list[i]]
    if editor_list[i] in list(editor_scores.keys()):  
      dataset.loc[i, 'Editor_score'] = editor_scores[editor_list[i]]
    if writer_list[i] in list(writer_scores.keys()):
      dataset.loc[i, 'Writer_score'] = writer_scores[writer_list[i]]
    if cast1[i] in list(cast1_score.keys()):  
      dataset.loc[i, 'Cast1'] = cast1_score[cast1[i]]
    if cast2[i] in list(cast2_score.keys()):
      dataset.loc[i, 'Cast2'] = cast2_score[cast2[i]]
      dataset.loc[i, 'Cast3'] = cast3_score[cast3[i]]
      dataset.loc[i, 'Cast4'] = cast4_score[cast4[i]]
      dataset.loc[i, 'Cast5'] = cast5_score[cast5[i]]

del dataset['homepage']
del dataset['id']
del dataset['keywords']
del dataset['overview']
del dataset['production_countries']
del dataset['status']
del dataset['tagline']
del dataset['title']
del dataset['genres']
del dataset['production_companies']
del dataset['release_date']
del dataset['spoken_languages']
del dataset['genre_binary_list']

dataset = dataset.join(pd.DataFrame({
        'Revenue': 0,
        }, index = mov.index))

dataset['Revenue'] = dataset['revenue']

del dataset['revenue']
del dataset['original_title']
#-----------------------------------------------------------------------------#

#---------------------------Linear Regression Begins--------------------------#

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# One Hot Encoding the languages
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x[:, 0] = labelencoder_x.fit_transform(x[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
x = onehotencoder.fit_transform(x).toarray()