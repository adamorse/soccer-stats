# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 09:20:32 2018

@author: ada morse

a neural net to predict home wins in soccer games

input data: 
            half-time goal difference (home goals - away goals)
            full-time on-target shot difference (home - away)

output:
            a neural net to answer the following question at half-time:
                if the game continues as it has (measured by shot difference),
                what is the probability of a home win?
"""

import pandas as pd
import glob

# the data files have already been acquired and cleaned

# build a filename recognizer
filenames = glob.glob('data/pred_winner/*.csv')

# initiate a list of DataFrames, one for each season file
season_data = []

# generate array of column labels of interest
# full-time home goals, full-time away goals
# half-time  home goals, half-time away goals
# home shots on target, away shots on target

cols = ['FTHG','FTAG','HTHG','HTAG','HST','AST']

for file in filenames:
    # read the file into a DataFrame season, selecting only the required columns 
    # and dropping any rows of all NaNs
    # 19 seasons don't have shots on target data, we'll skip those
    try:
        season = pd.read_csv(file,encoding = "ISO-8859-1",usecols=cols).dropna(axis=0,how='any')
        season_data.append(season)
    except ValueError:
        seasons_skipped += 1

# concatenate data into a single dataframe
        
data = pd.concat(season_data).reset_index(drop=True)

# add a column for win/draw/loss to the DataFrame
# this column will have 2 for a win, 1 for a draw, 0 for a loss

data['win_drawloss'] = (data.FTHG - data.FTAG > 0).astype(int)

# drop unnecessary columns
data = data.drop(['FTHG','FTAG'],axis=1)

# add a column for on-target shot difference
data['shots'] = data['HST'] - data['AST']

# add a column for half-time goal-difference
data['halftime'] = data['HTHG'] - data['HTAG']

# drop unnecessary columns
data = data.drop(['HST','AST','HTHG','HTAG'],axis=1)

# build the model

import numpy as np
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

# convert the win/draw-loss column to categories
target = to_categorical(data.win_drawloss)

# set up array of predictor variables
predictors = np.array(data.drop(['win_drawloss'],axis=1))

# Construct a sequential neural network model
neural_net = Sequential()

# Construct hidden layers
neural_net.add(Dense(32,activation='relu',input_shape=(predictors.shape[1],)))
neural_net.add(Dense(32,activation='relu'))

# Construct output layer
neural_net.add(Dense(2,activation='softmax'))

# Compile the model
neural_net.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

# Stop after
stop_early = EarlyStopping(patience=2)

# Fit the model
neural_net.fit(predictors,target,validation_split=.3,callbacks=[stop_early])

#save the model
neural_net.save('home_wins.h5')
