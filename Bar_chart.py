#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 22:35:30 2018

@author: naveen, Milan(urf one and only mIJo 2117 yo!)
"""

#Importing the Important Libraries
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

#For converting the string representation of 
#dict to dict
import ast

#Extracting the data from csv files
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

#The loop to get the maximum number of distinct genres
for index,i in zip(movies.index,movies['genres']):
    for j in range(len(i)):
        if i[j] not in list2:
            list2.append(i[j])
            
no_of_genres = len(list2)

#Forming the genre_binary_list
movies['genre_binary_list'] = movies.index

for index,i in zip(movies.index,movies['genres']):
    samlist = np.zeros(21,dtype=int);
    for j in range(len(list2)):
        if list2[j] in movies.loc[index,'genres']:
            samlist[j] = 1
    movies.loc[index,'genre_binary_list' ] = str(samlist)

movies['genre_binary_list']=movies['genre_binary_list'].str.strip('[]')
movies['genre_binary_list']=movies['genre_binary_list'].str.split(' ')

movie_director = {}
m = []

#Converting the dict-like-string to dict
for j in range(0,len(mov['crew'])):
    m.append( ast.literal_eval( mov['crew'][j]) )

#Forming a dictionary with key as the name of the 
#movie and value as director's name.    
for j in range(0,len(m)):
    for i in range(0,len(m[j])):
        if m[j][i]['job'] == "Director":
            movie_director[mov['title'][j]] = m[j][i]['name']
            #print(i)
            i = len(m[j])

#Director name and the number of movies he has 
#directed.
director_count = {}

for j in movie_director:
    if movie_director[j] not in list(director_count.keys()):
        director_count[movie_director[j]] = 1
    else:
        director_count[movie_director[j]] += 1

import operator


director_gross = {}

for j1 in range(0,len(movies['title'])):
    j = movies['title'][j1]
    if j in list(movie_director.keys()):
        if movie_director[j] not in list(director_gross.keys()):
           director_gross[movie_director[j]] = movies['revenue'][j1]
        else:
            director_gross[movie_director[j]] += movies['revenue'][j1]

director_gross = sorted(director_gross.items(), key=operator.itemgetter(1), reverse = True)

top_10_director = []
gross_only = [] 
name_only = []
for i in range(0,10):
    top_10_director.append(director_gross[i])
    gross_only.append( float(top_10_director[i][1]/1000000000) )
    name_only.append( top_10_director[i][0] )
    i += 1

gross_only = np.array(gross_only)
x_pos = []
i = 2
while True:
    x_pos.append(i)
    i += 5
    if i > 47:
        break
my_colors = ['#624ea7', 'g', 'yellow', 'maroon', 'maroon','#624ea7', 'g', 'yellow', 'maroon', 'maroon']
plt.bar(x_pos, gross_only, width = 4, colors = my_colors)
plt.xticks(x_pos, name_only)
plt.xticks(rotation=90)
plt.ylabel('Gross in billions')
plt.title('Gross of top Directors')




