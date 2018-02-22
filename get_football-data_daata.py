# code to get data from football-data.co.uk/mmz4281/

# Import packages
from urllib.request import urlretrieve
import pandas as pd

# base for file urls
url_base = 'http://www.football-data.co.uk/mmz4281/'
    
# list of years
years = [str((95 + i)%100) if len(str((95+i)%100))==2 
     else '0' + str((95 + i)%100) for i in range(0,23)]
    
# list of league seasons
seasons = [years[i]+years[i+1] for i in range(0,22)]
    
# list of leagues
leagues = ['E0', 'D1', 'I1', 'SP1', 'F1']
    
# list of urls and filenames to store locally
urls = [url_base + season + '/' + league + '.csv' 
    for season in seasons for league in leagues]
filenames = ['data/' + league + season + '.csv' for season in seasons for league in leagues]

def get_data(urls,filenames):
    ''' Function to grab data files from the web at urls and store locally in data/filenames '''    
    for idx,url in enumerate(urls):
        urlretrieve(url,filenames[idx])

 get_data()