# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 08:38:14 2018

@author: ada morse

compute how quickly soccer league tables converge to the final distribution
"""

import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.stats import entropy
from scipy.optimize import curve_fit

# function to compute Jensen-Shannon divergence

def JSD(p, q):
    r = 0.5 * (p + q)
    return 0.5 * (entropy(p, r) + entropy(q, r))

# the data files have already been acquired and cleaned
# see get_football-data_data.py

# build a list of filenames
filenames = glob.glob('data/*.csv')

# initialize an array to hold JSD values
# each row will contain the JSD curve data for one season
jsds = np.zeros((len(filenames),500))

# initialize an array to hold final league tables
finals = np.zeros((len(filenames),25))

# initialize a season counter
season = 0

# list of columns needed from the data files
cols = ['Date','HomeTeam','AwayTeam','FTHG','FTAG']

for file in filenames:
    # load the season data
    df = pd.read_csv(file,index_col='Date',encoding = "ISO-8859-1",usecols=cols).dropna(axis=0,how='any')
    
    # get the unique team names for that season
    teams = list(df.HomeTeam.unique())
    
    # set up array for league tables
    # each column corresponds to a team
    # each row corresponds to the league table after that number of games
    tables = np.zeros((df.shape[0]+1,len(teams)))
    
    # initialize game counter
    num_games = 1
    
    # loop through the season data game by game
    for idx,row in df.iterrows():
        # initialize the current league table to be the same as the last
        tables[num_games,:] = tables[num_games-1,:]
        
        # get indices for the teams involved in thisgame
        home_idx = teams.index(row['HomeTeam'])
        away_idx = teams.index(row['AwayTeam'])
        
        # compute home goals - away goals
        goal_diff = row.FTHG - row.FTAG
        
        # update the league table based on the result
        if goal_diff > 0:
            tables[num_games,home_idx] += 3
        elif goal_diff < 0:
            tables[num_games,away_idx] += 3
        else:
            tables[num_games,home_idx] += 1
            tables[num_games,away_idx] += 1
        
        # increment the game counter
        num_games += 1
    
    # delete first row of the table
    tables = tables[1:,:]
    
    # compute the probability distribution for the final league table
    p = tables[-1,:]/np.sum(tables[-1,:])
    
    # store p
    for idx,team in enumerate(p):
        finals[season,idx] = team
    
    # for each of the running league tables, convert to a distribution
    # and then compute the JSD
    for i in range(len(tables[:,0])):
        #if np.count_nonzero(tables[idx,:]) == len(tables[idx,:]):
        q = tables[i,:]/np.sum(tables[i,:])
        jsds[season,i] = JSD(p,q)
    
    # increment the season counter
    season += 1
    
# compute the average JSD curve    
avg = np.sum(jsds,axis=0)/110

# array of x values for the games
xs = np.array([i for i in range(len(avg))])

# define function for curve-fitting
def f(x, a, b, c):
     return a * np.exp(-b * x) + c

# perform the curve fit
popt, pcov = curve_fit(f, xs, avg)

# plot the individual JSD curves
for i in range(jsds.shape[0]):
    plt.plot(jsds[i,:],alpha=.3,color='gray')
    
# add title and axis labels
plt.title('Convergence of league tables over time')
plt.xlabel('Number of games played')
plt.ylabel('JSD with final table')

# set axis limits, 461 most games in an individual season
axes = plt.gca()
axes.set_xlim([0,461])

plt.savefig('allseasons.png')

# zoom in on the first 100 games
axes.set_xlim([0,100])
plt.savefig('convbegin.png')

# zoom out again
axes.set_xlim([0,380])

# plot the average curve
plt.plot(xs,avg,'b-',label='average JSD')

# add a legend
plt.legend()
plt.savefig('convwithavg.png')

# plot the best-fit curve
plt.plot(xs, f(xs, *popt), 'r-',
          label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))

# update the legend
plt.legend()
plt.savefig('conv.png')
plt.show()

plt.clf()
plt.cla()
plt.close()

# compute examples of final probability distributions

# spain 16-17
xd = [i for i in range(18)]
plt.bar(xd,np.sort(finals[5,:18]))
plt.title('La Liga 2016-2017')
plt.xticks([],'')
plt.xlabel('Ranked teams')
plt.ylabel('Point distribution')
plt.savefig('sp1617.png')
plt.clf()
plt.cla()
plt.close()

# italy 16-17
xd = [i for i in range(20)]
plt.bar(xd,np.sort(finals[27,:20]))
plt.title('Serie A 2016-2017')
plt.xticks([],'')
plt.xlabel('Ranked teams')
plt.ylabel('Point distribution')
plt.savefig('it1617.png')
plt.clf()
plt.cla()
plt.close()

# france 16-17
xd = [i for i in range(20)]
plt.bar(xd,np.sort(finals[49,:20]))
plt.title('Ligue 1 2016-2017')
plt.xticks([],'')
plt.xlabel('Ranked teams')
plt.ylabel('Point distribution')
plt.savefig('fr1617.png')
plt.clf()
plt.cla()
plt.close()

# england 16-17
xd = [i for i in range(20)]
plt.bar(xd,np.sort(finals[71,:20]))
plt.title('Premier League 2016-2017')
plt.xticks([],'')
plt.xlabel('Ranked teams')
plt.ylabel('Point distribution')
plt.savefig('en1617.png')
plt.clf()
plt.cla()
plt.close()

# germany 16-17
xd = [i for i in range(18)]
plt.bar(xd,np.sort(finals[93,:18]))
plt.title('Bundesliga 2016-2017')
plt.xticks([],'')
plt.xlabel('Ranked teams')
plt.ylabel('Point distribution')
plt.savefig('ge1617.png')
plt.clf()
plt.cla()
plt.close()

# generate animation
# code below based on an example by Jake Vanderplas: 
# email: vanderplas@astro.washington.edu
# website: http://jakevdp.github.com
# license: BSD

# set up the figure
fig = plt.figure()

# set up the axes
ax = plt.axes(xlim=(0, 20), ylim=(0, .12))
line, = ax.plot([], [],'o',linestyle='None')

# add title, legend, etc.
plt.title('\'00-\'01 Bundesliga points distribution over time')
plt.xticks([],'')
plt.xlabel('Ranked teams')
plt.ylabel('Proportion of total points')

# draw the background 
def init():
    line.set_data([],[])
    plt.bar([i for i in range(20)],np.sort(tables[-1,:]/np.sum(tables[-1,:])),alpha=.3)
    return line,

# animation function, each frame draws a distribution after one more game
def animate(i):
    xd = [i for i in range(20)]
    y = np.sort(tables[i+40,:]/np.sum(tables[i+40,:]))
    line.set_data(xd, y)
    return line,

# animate
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=340, interval=20, blit=True,repeat_delay=1000)

# save the animation
anim.save('basic_animation.mp4', fps=50, extra_args=['-vcodec', 'libx264'])

plt.show()