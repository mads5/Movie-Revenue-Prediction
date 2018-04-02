#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 22:35:30 2018

@author: naveen
"""
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')
import base64
import io

import codecs




import PIL
mpl.rcParams['font.size']=12                #10 
mpl.rcParams['savefig.dpi']=100             #72 
mpl.rcParams['figure.subplot.bottom']=.1

movies=pd.read_csv('tmdb_5000_movies.csv')
mov=pd.read_csv('tmdb_5000_credits.csv')

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

# list2 will contain unique genres
# all_genres will contain all the genres to create the wordcloud
list2 = []
all_genres = []

for index,i in zip(movies.index,movies['genres']):
    for j in range(len(i)):
        if i[j] not in list2:
            list2.append(i[j])
        all_genres.append(i[j])
            
no_of_genres = len(list2)
all_genres = str(all_genres)
all_genres = all_genres.strip('[]');
all_genres = all_genres.replace(' ','');
all_genres = all_genres.replace("'",'');
all_genres = all_genres.replace(",",' ');
all_genres = all_genres.replace("  ",' ');

# adding a new column which has a binary string 
movies['genre_binary_list'] = movies.index

for index,i in zip(movies.index,movies['genres']):
    samlist = np.zeros(21,dtype=int);
    for j in range(len(list2)):
        if list2[j] in movies.loc[index,'genres']:
            samlist[j] = 1
    movies.loc[index,'genre_binary_list' ] = str(samlist)

movies['genre_binary_list']=movies['genre_binary_list'].str.strip('[]')
movies['genre_binary_list']=movies['genre_binary_list'].str.split(' ')

# creating a word cloud for the genres
import random
"""def grey_color_func(word, font_size, position, orientation, random_state=None,
                    **kwargs):
    return "hsl(0, 0%%, %d%%)" % random.randint(60, 225)
def random_color_func(word=None, font_size=None, position=None,  orientation=None, font_path=None, random_state=None):
    h = int(100.0 * float(random_state.randint(, 200)) / 255.0)
    s = int(100.0 * float(random_state.randint(100, 200)) / 255.0)
    l = int(100.0 * float(random_state.randint(30, 100)) / 255.0)

    return "hsl({}, {}%, {}%)".format(h, s, l)"""

from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)
wordcloud = WordCloud(
                          max_font_size = 200,
                          background_color='white',
                          random_state=42,
                          collocations = False
                         ).generate(str(all_genres))
print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
fig.savefig("word1.png", dpi=1080)

# considering the cast column
all_actors = []
# json to string
mov['cast']=mov['cast'].apply(json.loads)
for index,i in zip(mov.index,mov['cast']):
    actor1 = []
    if len(i) > 0:
        actor1.append((i[0]['name']))# the key 'name' contains the name of the primary actor
        all_actors.append(i[0]['name'])
    mov.loc[index,'cast']=str(actor1)
# string to list
mov['cast']=mov['cast'].str.strip('[]')
mov['cast']=mov['cast'].str.replace("'",'')
mov['cast']=mov['cast'].str.split(',')

# word cloud for actors
from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)
wordcloud = WordCloud(
                          max_font_size = 200,
                          background_color='white',
                          random_state=42,
                          collocations = False
                         ).generate(str(all_actors))
print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
fig.savefig("word1.png", dpi=1080)

year_and_inflation = {}

year_and_inflation[2017] = 2.11
year_and_inflation[2016] = 2.07
year_and_inflation[2015] = 0.73
year_and_inflation[2014] = 0.76
year_and_inflation[2013] = 1.50
year_and_inflation[2012] = 1.74
year_and_inflation[2011] = 2.96
year_and_inflation[2010] = 1.50
year_and_inflation[2009] = 2.72
year_and_inflation[2008] = 0.09
year_and_inflation[2007] = 4.08
year_and_inflation[2006] = 2.11








