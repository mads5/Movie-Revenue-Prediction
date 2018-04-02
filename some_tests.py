# -*- coding: utf-8 -*-
'''
  Some Tests to see some relations between data 
  and visualize it accordingly for figuring out
  the roots of the problem in increasing the accuracy
  of the model that we are preparing.
  
  Author: Milan

'''

# Using Python to load a CSV file
import csv

with open('tmdb_5000_credits.csv') as movie_reader:
    
    # Delimiter is the value that sepeartes the rows of data
    data_1 = csv.reader(movie_reader, delimiter = ',')
    
    for row in data_1:
        print(row[0])
        print('\n')

# Using numpy to load CSV file
import pandas as pd
import numpy as np

dataframe = pd.read_csv('tmdb_5000_credits.csv')    