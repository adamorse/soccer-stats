# soccer-stats
a repository of soccer data analysis

most data used was pulled from football-data.co.uk
code to do this is in get_football-data_data.py

## project list

### home win predictor

neural net for "predicting" whether a game will end in a hometeam win based on the halftime goal difference and fulltime shot difference

trained on data from the top 5 European leagues

currently achieves ~77% accuracy (7th epoch: loss: 0.4700 - acc: 0.7656 - val_loss: 0.4519 - val_acc: 0.7748)
