#!/Library/Frameworks/Python.framework/Versions/3.8/bin/python3
# coding: utf-8

import pandas as pd
import numpy as np
import scipy.stats as stats
import re
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import Divider, Size
from mpl_toolkits.axes_grid1.mpl_axes import Axes
import warnings

# Set file directory
os.chdir(os.path.dirname(sys.argv[0]))

# Ignore pandas warnings
warnings.filterwarnings('ignore')

nhl_df=pd.read_csv("nhl.csv")
cities=pd.read_html("wikipedia_data.html")[1]
cities=cities.iloc[:-1,[0,3,5,6,7,8]]

def nhl_correlation():
    # YOUR CODE HERE
#     raise NotImplementedError()
    global nhl_df
    global cities
    # Make sure we can only use 2018 data in our answer
    nhl_df = nhl_df[nhl_df['year'] == 2018]

    # Replacing weird nan character with REAL AMERICAN NUMPY NaNs
    weirdchar = cities.iloc[50,5]
    # nhl_df.head()
    cities[cities == weirdchar] = np.nan

    ## View and clean up data
    # Wikipedia

    # Rename the metropolitan area column
    cities.rename(columns = {'Metropolitan area':'metro','Population (2016 est.)[8]':'pop'},inplace = True)

    # Clean up team names - Remove footnotes from team names
    def uncite(x):
    #     print("hi")
        t = pd.isnull(x['NHL'])
        if not t:
            test = re.sub('\[[\w ]+\]','', x['NHL'])
            return(test)
    # len((cities.apply(uncite, axis=1)).dropna()) == len(cities['NHL'].dropna())
    cities['NHL'] = cities.apply(uncite, axis=1)

    # First - Remove NaNs from cities data
    cities.dropna(subset = ['NHL'],inplace = True)

    # Clean up team names - split and make into list - WITHOUT ADDING AN EMPTY STRING IN TWO-WORD TEAM NAMES
    # AND ALSO MAKE SURE THAT WE DON'T SPLIT TWO WORD TEAM NAMES UP
    def splitter(x):
        t = pd.isnull(x['NHL'])
        if not t:
            if ' ' in x['NHL']:
                test = x['NHL'].strip()
                test = test.split(' ')[-1]
                return [test]
            else:
                test = re.sub('([a-z])([A-Z])', '\\1 \\2', x['NHL'])
                test = test.strip()
                test = test.split(' ')
                return test
    cities['NHL'] = cities.apply(splitter, axis = 1)

    # Make pop numeric
    cities['pop'] = cities['pop'].astype('int64')

    # Now drop all entries where citation was the only element
    l = []
    for x in cities.index:
        if ('' in cities['NHL'][x]):
            l.append(x)
    # print(l)
    cities.drop(l,inplace = True)

    ## NHL
    # Remove asterisks from team names
    nhl_df['team'] = nhl_df.apply(lambda x: re.sub('[*]','',x['team']), axis=1)

    # Remove rows containing 'Division' using regex
    divline = 0 # Used to change column holding 'Division' depending on the dataset (I just found out this isn't needed)
    nhl_df['Div?'] = nhl_df.apply(lambda x: pd.isnull(re.search('Division',x.iloc[divline])), axis = 1)
    nhl_df['Div?'] = ~nhl_df['Div?'] # Need to flip bool values to True - couldn't think of a way to do it above

    nhl_df.drop(nhl_df[nhl_df['Div?']].index, inplace = True)

    # Make sure 'W' and 'L' are numeric data
    nhl_df['W'] = nhl_df['W'].astype('int64')
    nhl_df['L'] = nhl_df['L'].astype('int64')

    ## Attach cities to nhl dataset
    # Create team name column
    nhl_df['team_short'] = nhl_df.apply(lambda x: x['team'].split(' ')[-1],axis = 1)

    ## Attach cities to nhl dataset
    # Now link the datasets by team name
    l = []
    for i in nhl_df['team_short']:
    #     print('-------BEGIN: ' + i + '--------')
        for n in cities['NHL']:
            if i in n:
    #             print(i,n)
                l.append(tuple([i,n]))
    l = (dict(l))

    # if len(l) == len(nhl_df):
    #     print('Success! Merged lists equal.')
    # print(l)

    nhl_df['NHL'] = nhl_df.apply(lambda x: l[x['team_short']],axis = 1)
    nhl_df['NHL'] = nhl_df.apply(lambda x: tuple(x['NHL']),axis=1)
    cities['NHL'] = cities.apply(lambda x: tuple(x['NHL']),axis=1)
    nhl_df = pd.merge(nhl_df,cities,how= 'left',on='NHL')

    ## Calculate W/L ratios for each team
    def win_loss(x):
        return x['W'] / (x['W'] + x['L'])

    nhl_df['W/L%'] = nhl_df.apply(win_loss, axis = 1)

    ## Average W/L ratios for each city's teams and find pop by city
    win_loss_by_region = nhl_df.groupby('metro').agg({'W/L%':np.mean})
    population_by_region = cities[['metro','pop']]
    population_by_region.set_index('metro',inplace=True)
    merged = pd.merge(population_by_region,win_loss_by_region,left_index=True,right_index=True)

    assert len(population_by_region) == len(win_loss_by_region), "Q1: Your lists must be the same length"
    assert len(population_by_region) == 28, "Q1: There should be 28 teams being analysed for NHL"

#     return stats.pearsonr(merged['W/L%'],merged['pop'])[0]
    return merged


# In[50]:




df = nhl_correlation()

def pop_wl_plot(data,league):

    fig = plt.figure(figsize=(9, 6))

    # The first items are for padding and the second items are for the axes.
    # sizes are in inch.
    h = [Size.Fixed(1.5), Size.Fixed(6)]
    v = [Size.Fixed(.5), Size.Fixed(5.)]

    divider = Divider(fig, (0.0, 0.0, 0., 0.), h, v, aspect=False)
    # the width and height of the rectangle is ignored.

    ax = Axes(fig, divider.get_position())
    ax.set_axes_locator(divider.new_locator(nx=1, ny=1))

    fig.add_axes(ax)

    data1 = list(data['pop']/1000000)
    data2 = list(data['W/L%'])
    lab = list(data.index,)

    # data2

    ax.set_ylabel('Win/Loss Percentage')
    ax.set_xlabel('Population (Millions)')
    plt.title(
        'Effects of Corresponding Metropolitan Area Population \n on ' + league + ' Team Success')
    plt.scatter(data1,data2)
    # plt.yticks(ticks=list(np.arange(0,1.05,.1)))

    for i, txt in enumerate(lab):
    #     print(i)
    #     print(txt)
        ax.annotate(txt, (data1[i],data2[i]),(data1[i]+.25,data2[i]+.005),rotation =25)

    # Add a horizontal line
    plt.plot([0,20],[.5,.5],linestyle = '--',color = 'orange')

    plt.show()

pop_wl_plot(df, 'NHL')

    # Stretch out figure to make labels more clear


    # print(len(data1))
    # print(len(data2))
    # print(len(lab))


nba_df=pd.read_csv("nba.csv")
cities=pd.read_html("wikipedia_data.html")[1]
cities=cities.iloc[:-1,[0,3,5,6,7,8]]

def nba_correlation():
    # YOUR CODE HERE
#     raise NotImplementedError()
    global nba_df
    global cities


    ## The following function will clean up the
    # Wikipedia
    def clean_wiki(league):
        # Replacing weird nan character with REAL AMERICAN NUMPY NaNs
        weirdchar = cities.iloc[50,5]
        # nhl_df.head()
        cities[cities == weirdchar] = np.nan

        # Rename the metropolitan area column
        cities.rename(columns = {'Metropolitan area':'metro','Population (2016 est.)[8]':'pop'},inplace = True)

        # Clean up team names - Remove footnotes from team names
        def uncite(x):
        #     print("hi")
            t = pd.isnull(x[league])
            if not t:
                test = re.sub('\[[\w ]+\]','', x[league])
                return(test)
        # len((cities.apply(uncite, axis=1)).dropna()) == len(cities['NHL'].dropna())
        cities[league] = cities.apply(uncite, axis=1)

        # First - Remove NaNs from cities data
        cities.dropna(subset = [league],inplace = True)

        # Clean up team names - split and make into list - WITHOUT ADDING AN EMPTY STRING IN TWO-WORD TEAM NAMES
        # AND ALSO MAKE SURE THAT WE DON'T SPLIT TWO WORD TEAM NAMES UP
        def splitter(x):
            t = pd.isnull(x[league])
            if not t:
                if ' ' in x[league]:
                    test = x[league].strip()
                    test = test.split(' ')[-1]
                    return [test]
                else:
                    test = re.sub('([a-z])([A-Z])', '\\1 \\2', x[league])
                    test = test.strip()
                    test = test.split(' ')
                    return test
        cities[league] = cities.apply(splitter, axis = 1)

        # Make pop numeric
        cities['pop'] = cities['pop'].astype('int64')

        # Now drop all entries where citation was the only element
        l = []
        for x in cities.index:
            if ('' in cities[league][x]):
                l.append(x)
        # print(l)
        cities.drop(l,inplace = True)
        return cities
    cities = clean_wiki(league='NBA')

    ## Format nba
    # Slice only 2018 data
    nba_df = nba_df[nba_df['year'] == 2018]

    # format W/L% as a float64
    nba_df['W/L%'] = nba_df['W/L%'].astype('float64')

    # Remove asterisks from team names
    nba_df['team'] = nba_df.apply(lambda x: re.sub('[*]','',x['team']), axis=1)

    # Remove parentheses
    nba_df['team'] = nba_df.apply(lambda x: re.sub('\\xa0\(\d+\)','',x['team']), axis=1)

    ## Attach cities to nhl dataset
    def merge_datasets(league_df, league):

        # Create team name column
        league_df['team_short'] = league_df.apply(lambda x: x['team'].split(' ')[-1],axis = 1)

        ## Attach cities to nhl dataset
        # Now link the datasets by team name
        l = []
        for i in league_df['team_short']:
        #     print('-------BEGIN: ' + i + '--------')
            for n in cities[league]:
                if i in n:
        #             print(i,n)
                    l.append(tuple([i,n]))
        l = (dict(l))

        # if len(l) == len(nhl_df):
        #     print('Success! Merged lists equal.')
        # print(l)

        league_df[league] = league_df.apply(lambda x: l[x['team_short']],axis = 1)
        league_df[league] = league_df.apply(lambda x: tuple(x[league]),axis=1)
        cities[league] = cities.apply(lambda x: tuple(x[league]),axis=1)
        league_df = pd.merge(league_df,cities,how= 'left',on=league)
        return league_df
    nba_df = merge_datasets(league_df=nba_df,league = 'NBA')

    win_loss_by_region = nba_df.groupby('metro').agg({'W/L%':np.mean})
    population_by_region = cities[['metro','pop']]
    population_by_region.set_index('metro',inplace=True)
    merged = pd.merge(population_by_region,win_loss_by_region,left_index=True,right_index=True)

    # population_by_region = [] # pass in metropolitan area population from cities
    # win_loss_by_region = [] # pass in win/loss ratio from nba_df in the same order as cities["Metropolitan area"]

    assert len(population_by_region) == len(win_loss_by_region), "Q2: Your lists must be the same length"
    assert len(population_by_region) == 28, "Q2: There should be 28 teams being analysed for NBA"

#     return(stats.pearsonr(merged['W/L%'],merged['pop']))
    return merged


# In[56]:


df = nba_correlation()


# In[57]:


pop_wl_plot(df, 'NBA')

mlb_df=pd.read_csv("mlb.csv")
cities=pd.read_html("wikipedia_data.html")[1]
cities=cities.iloc[:-1,[0,3,5,6,7,8]]

def mlb_correlation():
    # YOUR CODE HERE
#     raise NotImplementedError()
    global cities
    global mlb_df
    def clean_wiki(league):
        # Replacing weird nan character with REAL AMERICAN NUMPY NaNs
        weirdchar = cities.iloc[50,5]
        # nhl_df.head()
        cities[cities == weirdchar] = np.nan

        # Rename the metropolitan area column
        cities.rename(columns = {'Metropolitan area':'metro','Population (2016 est.)[8]':'pop'},inplace = True)

        # Clean up team names - Remove footnotes from team names
        def uncite(x):
        #     print("hi")
            t = pd.isnull(x[league])
            if not t:
                test = re.sub('\[[\w ]+\]','', x[league])
                return(test)
        # len((cities.apply(uncite, axis=1)).dropna()) == len(cities['NHL'].dropna())
        cities[league] = cities.apply(uncite, axis=1)

        # First - Remove NaNs from cities data
        cities.dropna(subset = [league],inplace = True)

        #Split beforehand
        cities[league] = cities.apply(lambda x: x[league].strip(), axis = 1)

        # Clean up team names - split and make into list - WITHOUT ADDING AN EMPTY STRING IN TWO-WORD TEAM NAMES
        # AND ALSO MAKE SURE THAT WE DON'T SPLIT TWO WORD TEAM NAMES UP
        def splitter(x):
            t = pd.isnull(x[league])
    #         print(x[league])
            if not t:
                if ' ' in x[league]:
                    if pd.isnull(re.search('[a-z][A-Z]', x[league])):
                        test = x[league]
                        test = test.strip()
                        return [test.split(' ')[-1]]
                    else:
                        if (re.search(' ', x[league])).start()>(re.search('[a-z][A-Z]', x[league])).start():
                            test = x[league].strip()
                            test1 = test.split(' ')[-1]
                            test2 = test.split(' ')[-2]
                            test2 = re.sub('([a-z])([A-Z])', '\\1 \\2', test2)
                            test2 = test2.strip()
                            test2 = test2.split(' ')[0]
                            test = [test1,test2]
                            return [test1,test2]
                        else:
                            test = x[league].strip()
                            test = test.split(' ')
                            test = test.split(' ')[-1]
                            test = re.sub('([a-z])([A-Z])', '\\1 \\2', test)
                            test = test.strip()
                            test1 = test.split(' ')[0]
                            test2 = test.split(' ')[1]
                            test = [test1,test2]
                else:
                    test = re.sub('([a-z])([A-Z])', '\\1 \\2', x[league])
                    test = test.strip()
                    test = test.split(' ')
                    return test
        cities[league] = cities.apply(splitter, axis = 1)

        # Make pop numeric
        cities['pop'] = cities['pop'].astype('int64')

        # Now drop all entries where citation was the only element
        l = []
        for x in cities.index:
            if ('' in cities[league][x]):
                l.append(x)
        # print(l)
        cities.drop(l,inplace = True)
        return cities
    cities = clean_wiki(league='MLB')

    ## Format mlb
    def clean_league(dataframe,wl_col):
        # Slice only 2018 data
        dataframe = dataframe[dataframe['year'] == 2018]

        dataframe.rename(columns = {wl_col:'W/L%'},inplace=True)

        # format W/L% as a float64
        dataframe['W/L%'] = dataframe['W/L%'].astype('float64')

        # Remove asterisks from team names
        dataframe['team'] = dataframe.apply(lambda x: re.sub('[*]','',x['team']), axis=1)

        # Remove parentheses
        dataframe['team'] = dataframe.apply(lambda x: re.sub('\\xa0\(\d+\)','',x['team']), axis=1)
        return dataframe
    mlb_df = clean_league(mlb_df,'W-L%')

    ## Attach cities to nhl dataset
    def merge_datasets(league_df, league):

        # Create team name column
        league_df['team_short'] = league_df.apply(lambda x: x['team'].split(' ')[-1],axis = 1)

        ## Attach cities to nhl dataset
        # Now link the datasets by team name
        l = []
        for i in league_df['team_short']:
        #     print('-------BEGIN: ' + i + '--------')
            for n in cities[league]:
                if i in n:
        #             print(i,n)
                    l.append(tuple([i,n]))
        l = (dict(l))

        # if len(l) == len(nhl_df):
        #     print('Success! Merged lists equal.')
        # print(l)

        league_df[league] = league_df.apply(lambda x: l[x['team_short']],axis = 1)
        league_df[league] = league_df.apply(lambda x: tuple(x[league]),axis=1)
        cities[league] = cities.apply(lambda x: tuple(x[league]),axis=1)
        league_df = pd.merge(league_df,cities,how= 'left',on=league)
        return league_df
    mlb_df = merge_datasets(league_df=mlb_df,league = 'MLB')

    win_loss_by_region = mlb_df.groupby('metro').agg({'W/L%':np.mean})
    population_by_region = cities[['metro','pop']]
    population_by_region.set_index('metro',inplace=True)
    merged = pd.merge(population_by_region,win_loss_by_region,left_index=True,right_index=True)

    assert len(population_by_region) == len(win_loss_by_region), "Q3: Your lists must be the same length"
    assert len(population_by_region) == 26, "Q3: There should be 26 teams being analysed for MLB"

#     return(stats.pearsonr(merged['W/L%'],merged['pop']))
    return merged


df = mlb_correlation()

pop_wl_plot(df, 'MLB')

nfl_df=pd.read_csv("nfl.csv")
cities=pd.read_html("wikipedia_data.html")[1]
cities=cities.iloc[:-1,[0,3,5,6,7,8]]

def nfl_correlation():
    # YOUR CODE HERE
#     raise NotImplementedError()

    global cities
    global nfl_df
    def clean_wiki(league):

            # Replacing weird nan character with REAL AMERICAN NUMPY NaNs
        weirdchar = cities.iloc[50,5]
        # nhl_df.head()
        cities[league] = cities.apply(lambda x: re.sub(weirdchar,'',x[league]),axis=1)
    #     cities[cities == weirdchar] = np.nan

        # Clean up team names - Remove footnotes from team names
        def uncite(x):
        #     print("hi")
            t = pd.isnull(x[league])
            if not t:
                test = re.sub('\[[\w ]+\]','', x[league])
                return(test)
        # len((cities.apply(uncite, axis=1)).dropna()) == len(cities['NHL'].dropna())
        cities[league] = cities.apply(uncite, axis=1)

        # Rename the metropolitan area column
        cities.rename(columns = {'Metropolitan area':'metro','Population (2016 est.)[8]':'pop'},inplace = True)

        # First - Remove NaNs from cities data
        cities.dropna(subset = [league],inplace = True)

        #Split beforehand
        cities[league] = cities.apply(lambda x: x[league].strip(), axis = 1)

        # Clean up team names - split and make into list - WITHOUT ADDING AN EMPTY STRING IN TWO-WORD TEAM NAMES
        # AND ALSO MAKE SURE THAT WE DON'T SPLIT TWO WORD TEAM NAMES UP
        def splitter(x):
            t = pd.isnull(x[league])
    #         print(x[league])
            if not t:
                if ' ' in x[league]:
                    if pd.isnull(re.search('[a-z][A-Z]', x[league])):
                        test = x[league]
                        test = test.strip()
                        return [test.split(' ')[-1]]
                    else:
                        if (re.search(' ', x[league])).start()>(re.search('[a-z][A-Z]', x[league])).start():
                            test = x[league].strip()
                            test1 = test.split(' ')[-1]
                            test2 = test.split(' ')[-2]
                            test2 = re.sub('([a-z])([A-Z])', '\\1 \\2', test2)
                            test2 = test2.strip()
                            test2 = test2.split(' ')[0]
                            test = [test1,test2]
                            return [test1,test2]
                        else:
                            test = x[league].strip()
                            test = test.split(' ')
                            test = test.split(' ')[-1]
                            test = re.sub('([a-z])([A-Z])', '\\1 \\2', test)
                            test = test.strip()
                            test1 = test.split(' ')[0]
                            test2 = test.split(' ')[1]
                            test = [test1,test2]
                else:
                    test = re.sub('([a-z])([A-Z])', '\\1 \\2', x[league])
                    test = test.strip()
                    test = test.split(' ')
                    return test
        cities[league] = cities.apply(splitter, axis = 1)

        # Make pop numeric
        cities['pop'] = cities['pop'].astype('int64')

        # Now drop all entries where citation was the only element
        l = []
        for x in cities.index:
            if ('' in cities[league][x]):
                l.append(x)
        # print(l)
        cities.drop(l,inplace = True)
        return cities
    cities = clean_wiki(league='NFL')

    ## Format NFL
    def clean_league(dataframe,wl_col):
        # Slice only 2018 data
        dataframe = dataframe[dataframe['year'] == 2018]

        # Remove rows containing 'Division' using regex
        divline = 0 # Used to change column holding 'Division' depending on the dataset
        dataframe['Div?'] = dataframe.apply(lambda x: pd.isnull(re.search('(AFC|NFC)',x.iloc[divline])), axis = 1)
        dataframe['Div?'] = ~dataframe['Div?'] # Need to flip bool values to True -
#         couldn't think of a way to do it above
        dataframe.drop(dataframe[dataframe['Div?']].index, inplace = True)

        dataframe.rename(columns = {wl_col:'W/L%'},inplace=True)

        # format W/L% as a float64
        dataframe['W/L%'] = dataframe['W/L%'].astype('float64')

        # Remove asterisks from team names
        dataframe['team'] = dataframe.apply(lambda x: re.sub('([*]|[+])','',x['team']), axis=1)

        # Remove parentheses
        dataframe['team'] = dataframe.apply(lambda x: re.sub('\\xa0\(\d+\)','',x['team']), axis=1)
        return dataframe
    nfl_df = clean_league(nfl_df,'W-L%')

    ## Attach cities to nhl dataset
    def merge_datasets(league_df, league):

        # Create team name column
        league_df['team_short'] = league_df.apply(lambda x: x['team'].split(' ')[-1],axis = 1)

        ## Attach cities to nhl dataset
        # Now link the datasets by team name
        l = []
        for i in league_df['team_short']:
        #     print('-------BEGIN: ' + i + '--------')
            for n in cities[league]:
                if i in n:
        #             print(i,n)
                    l.append(tuple([i,n]))
        l = (dict(l))

        # if len(l) == len(nhl_df):
        #     print('Success! Merged lists equal.')
        # print(l)

        league_df[league] = league_df.apply(lambda x: l[x['team_short']],axis = 1)
        league_df[league] = league_df.apply(lambda x: tuple(x[league]),axis=1)
        cities[league] = cities.apply(lambda x: tuple(x[league]),axis=1)
        league_df = pd.merge(league_df,cities,how= 'left',on=league)
        return league_df
    nfl_df = merge_datasets(league_df=nfl_df,league = 'NFL')

    win_loss_by_region = nfl_df.groupby('metro').agg({'W/L%':np.mean})
    population_by_region = cities[['metro','pop']]
    population_by_region.set_index('metro',inplace=True)
    merged = pd.merge(population_by_region,win_loss_by_region,left_index=True,right_index=True)

    assert len(population_by_region) == len(win_loss_by_region), "Q4: Your lists must be the same length"
    assert len(population_by_region) == 29, "Q4: There should be 29 teams being analysed for NFL"

#     return(stats.pearsonr(merged['W/L%'],merged['pop']))
    return merged


# In[67]:


df = nfl_correlation()


# In[68]:


pop_wl_plot(df, 'NFL')


mlb_df=pd.read_csv("mlb.csv")
nhl_df=pd.read_csv("nhl.csv")
nba_df=pd.read_csv("nba.csv")
nfl_df=pd.read_csv("nfl.csv")
cities=pd.read_html("wikipedia_data.html")[1]
cities=cities.iloc[:-1,[0,3,5,6,7,8]]


def sports_team_performance():
    # YOUR CODE HERE
#     raise NotImplementedError()

#     global mlb_df
#     global nhl_df
#     global nba_df
#     global nfl_df
#     global cities
    global p_values

    def clean_wiki(league):
        global cities
        cities=pd.read_html("wikipedia_data.html")[1]
        cities=cities.iloc[:-1,[0,3,5,6,7,8]]


        # ------------------BEGIN PASTED CODE------------------------------------------------------------
        # Replacing weird nan character with REAL AMERICAN NUMPY NaNs
        weirdchar = cities.iloc[50,5]
        # nhl_df.head()
        cities[league] = cities.apply(lambda x: re.sub(weirdchar,'',x[league]),axis=1)
        #     cities[cities == weirdchar] = np.nan

        # Clean up team names - Remove footnotes from team names
        def uncite(x):
        #     print("hi")
            t = pd.isnull(x[league])
            if not t:
                test = re.sub('\[[\w ]+\]','', x[league])
                return(test)
        # len((cities.apply(uncite, axis=1)).dropna()) == len(cities['NHL'].dropna())
        cities[league] = cities.apply(uncite, axis=1)

        # Rename the metropolitan area column
        cities.rename(columns = {'Metropolitan area':'metro','Population (2016 est.)[8]':'pop'},inplace = True)

        # First - Remove NaNs from cities data
        cities.dropna(subset = [league],inplace = True)

        #Strip beforehand
        cities[league] = cities.apply(lambda x: x[league].strip(), axis = 1)

        def nodup(x, league):
            t = pd.isnull(x[league])
        #         print(x[league])
            if not t:
                if ' ' in x[league]:
                    if pd.isnull(re.search('[a-z][A-Z]', x[league])):
                        test = x[league]
                        test = test.strip()
                        test = test.split(' ')[-1]
                        return [test]
                    else:
                        if (re.search(' ', x[league])).start()>(re.search('[a-z][A-Z]', x[league])).start():
                            test = x[league].strip()
                            test1 = test.split(' ')[-1]
                            test2 = test.split(' ')[-2]
                            test2 = re.sub('([a-z])([A-Z])', '\\1 \\2', test2)
                            test2 = test2.strip()
                            test2 = test2.split(' ')[0]
                            test = [test1,test2]
                            return [test1,test2]
                        else:
                            test = x[league].strip()
                            test = test.split(' ')
                            test = test.split(' ')[-1]
                            test = re.sub('([a-z])([A-Z])', '\\1 \\2', test)
                            test = test.strip()
                            test1 = test.split(' ')[0]
                            test2 = test.split(' ')[1]
                            test = [test1,test2]
                else:
                    test = re.sub('([a-z])([A-Z])', '\\1 \\2', x[league])
                    test = test.strip()
                    test = test.split(' ')
                    return test
        cities['no_dupes'] = cities.apply(lambda x: nodup(x, league), axis = 1)

        l = []
        for x in cities.index:
            if ('' in cities['no_dupes'][x]):
                l.append(x)
        # print(l)
        cities.drop(l,inplace = True)

        l = []
        # cities.apply(lambda x: x['no_dupes']), axis = 1)
        for x in cities['no_dupes']:
            l.extend(x)


        # cities
        short_list = pd.Series(l)
        ser = (short_list.value_counts() > 1)
        if 2 > len(ser[ser == True]) > 0: # NOTE: This code only works if there is 0 or 1 examples of shared team names
            val = ser[ser == True].index[0]
            cities[league] = cities.apply(lambda x: re.sub(val,'',x[league]),axis = 1)

        # ------------------END PASTED CODE------------------------------------------------------------

        #Strip beforehand
        cities[league] = cities.apply(lambda x: x[league].strip(), axis = 1)

        # Clean up team names - split and make into list - WITHOUT ADDING AN EMPTY STRING IN TWO-WORD TEAM NAMES
        # AND ALSO MAKE SURE THAT WE DON'T SPLIT TWO WORD TEAM NAMES UP
        def splitter(x):
            t = pd.isnull(x[league])
    #         print(x[league])
            if not t:
                if ' ' in x[league]:
                    if pd.isnull(re.search('[a-z][A-Z]', x[league])):
                        test = x[league]
                        test = test.strip()
                        return [test.split(' ')[-1]]
                    else:
                        if (re.search(' ', x[league])).start()>(re.search('[a-z][A-Z]', x[league])).start():
                            test = x[league].strip()
                            test1 = test.split(' ')[-1]
                            test2 = test.split(' ')[-2]
                            test2 = re.sub('([a-z])([A-Z])', '\\1 \\2', test2)
                            test2 = test2.strip()
                            test2 = test2.split(' ')[0]
                            test = [test1,test2]
                            return [test1,test2]
                        else:
                            test = x[league].strip()
                            test = test.split(' ')
                            test = test.split(' ')[-1]
                            test = re.sub('([a-z])([A-Z])', '\\1 \\2', test)
                            test = test.strip()
                            test1 = test.split(' ')[0]
                            test2 = test.split(' ')[1]
                            test = [test1,test2]
                else:
                    test = re.sub('([a-z])([A-Z])', '\\1 \\2', x[league])
                    test = test.strip()
                    test = test.split(' ')
                    return test
        cities[league] = cities.apply(splitter, axis = 1)

        # Make pop numeric
        cities['pop'] = cities['pop'].astype('int64')

        # Now drop all entries where citation was the only element
        l = []
        for x in cities.index:
            if ('' in cities[league][x]):
                l.append(x)
        # print(l)
        cities.drop(l,inplace = True)
        return cities

    ## Format league
    def clean_league(dataframe,wl_col):
        # Slice only 2018 data
        dataframe = dataframe[dataframe['year'] == 2018]

        # Remove rows containing 'AFC or NFC' using regex - for NFL dataset
        divline = 0 # Used to change column holding 'Division' depending on the dataset
        dataframe['conf'] = dataframe.apply(lambda x: pd.isnull(re.search('(AFC|NFC)',x.iloc[divline])), axis = 1)
        dataframe['conf'] = ~dataframe['conf'] # Need to flip bool values to True - couldn't think of a way to do it above
        dataframe.drop(dataframe[dataframe['conf']].index, inplace = True)

        # Remove rows containing 'Division' using regex - for NHL dataset
        divline = 0 # Used to change column holding 'Division' depending on the dataset
        dataframe['Div?'] = dataframe.apply(lambda x: pd.isnull(re.search('Division',x.iloc[divline])), axis = 1)
        dataframe['Div?'] = ~dataframe['Div?'] # Need to flip bool values to True - couldn't think of a way to do it above
        dataframe.drop(dataframe[dataframe['Div?']].index, inplace = True)

        dataframe.rename(columns = {wl_col:'W/L%'},inplace=True)

        # format W/L% as a float64
        try:
            dataframe['W/L%'] = dataframe['W/L%'].astype('float64')
        except:
            print("")
            # continue

        # Remove asterisks from team names
        dataframe['team'] = dataframe.apply(lambda x: re.sub('([*]|[+])','',x['team']), axis=1)

        # Remove parentheses
        dataframe['team'] = dataframe.apply(lambda x: re.sub('\\xa0\(\d+\)','',x['team']), axis=1)
        return dataframe

    ## Attach cities to nhl dataset
    def merge_datasets(league_df, league):

        # -----------------------BEGIN PASTED CODE HERE---------------------------------------------------
        # Erasing duplicate team_short values
        short_list = league_df.apply(lambda x: x['team'].split(' ')[-1],axis = 1)
        ser = (short_list.value_counts() > 1)
        if 2 > len(ser[ser == True]) > 0: # NOTE: This code only works if there is 0 or 1 examples of shared team names
            val = ser[ser == True].index[0]
            ind = league_df[short_list == val].index
            for c in range(0,len(league_df.columns)):
                if league_df.columns[c] == 'team':
                    col = c
                    break
            for t in ind:
                league_df.iloc[t,c] = re.sub(val,'', league_df.iloc[t,c]).strip()
        # -----------------------END PASTED CODE HERE---------------------------------------------------

        # Create team name column
        league_df['team_short'] = league_df.apply(lambda x: x['team'].split(' ')[-1],axis = 1)

        ## Attach cities to nhl dataset
        # Now link the datasets by team name
        l = []
        for i in league_df['team_short']:
        #     print('-------BEGIN: ' + i + '--------')
            for n in cities[league]:
                if i in n:
        #             print(i,n)
                    l.append(tuple([i,n]))
        l = (dict(l))

        # if len(l) == len(nhl_df):
        #     print('Success! Merged lists equal.')
        # print(l)

        league_df[league] = league_df.apply(lambda x: l[x['team_short']],axis = 1)
        league_df[league] = league_df.apply(lambda x: tuple(x[league]),axis=1)
        cities[league] = cities.apply(lambda x: tuple(x[league]),axis=1)
        league_df = pd.merge(league_df,cities,how= 'left',on=league)
        return league_df

    # # This is so that I only need to pass in one argument to the following functions
    # tup_nfl = (nfl_df,'NFL')
    # tup_nba = (nba_df,'NBA')
    # tup_nhl = (nhl_df,'NHL')
    # tup_mlb = (mlb_df,'MLB')

    # Note: p_values is a full dataframe, so df.loc["NFL","NBA"] should be the same as df.loc["NBA","NFL"] and
    # df.loc["NFL","NFL"] should return np.nan
    sports = ['NFL', 'NBA', 'NHL', 'MLB']
    p_values = pd.DataFrame({k:np.nan for k in sports}, index=sports)

    ## Function
    def league_comp(league1_df, league1, league1_wl, league2_df, league2, league2_wl):
        # print('RUNNING: ' + league1 + ' on ' + league2)
        # Re-load data
        global mlb_df
        global nhl_df
        global nba_df
        global nfl_df
        global cities
        global p_values

        mlb_df=pd.read_csv("mlb.csv")
        nhl_df=pd.read_csv("nhl.csv")
        nba_df=pd.read_csv("nba.csv")
        nfl_df=pd.read_csv("nfl.csv")
        cities=pd.read_html("wikipedia_data.html")[1]
        cities=cities.iloc[:-1,[0,3,5,6,7,8]]

        #Clean both leagues
        league1_df = clean_league(league1_df, league1_wl)
        league2_df = clean_league(league2_df, league2_wl)

    #     THE FOLLOWING CODE WAS MEANT TO ONLY CALCULATE W/L FOR NHL DATASET ---------------
    #     if 'W/L%' not in league1_df.columns:
    #         # Make sure 'W' and 'L' are numeric data
    #         league1_df['W'] = league1_df['W'].astype('int64')
    #         league1_df['L'] = league1_df['L'].astype('int64')

    #         ## Calculate W/L ratios for each team
    #         def win_loss(x):
    #             return x['W'] / (x['W'] + x['L'])

    #         league1_df['W/L%'] = league1_df.apply(win_loss, axis = 1)

    #     if 'W/L%' not in league2_df.columns:
    #         # Make sure 'W' and 'L' are numeric data
    #         league2_df['W'] = league2_df['W'].astype('int64')
    #         league2_df['L'] = league2_df['L'].astype('int64')

    #         ## Calculate W/L ratios for each team
    #         def win_loss(x):
    #             return x['W'] / (x['W'] + x['L'])

    #         league2_df['W/L%'] = league2_df.apply(win_loss, axis = 1)
    #     ----------------------------------------------------------------------------------------


        # Make sure 'W' and 'L' are numeric data
        league1_df['W'] = league1_df['W'].astype('int64')
        league1_df['L'] = league1_df['L'].astype('int64')

        ## Calculate W/L ratios for each team
        def win_loss(x):
            return x['W'] / (x['W'] + x['L'])

        league1_df['W/L%'] = league1_df.apply(win_loss, axis = 1)

        # Make sure 'W' and 'L' are numeric data
        league2_df['W'] = league2_df['W'].astype('int64')
        league2_df['L'] = league2_df['L'].astype('int64')

        ## Calculate W/L ratios for each team
        def win_loss(x):
            return x['W'] / (x['W'] + x['L'])

        league2_df['W/L%'] = league2_df.apply(win_loss, axis = 1)


        # Merge leagues with wiki
        cities = clean_wiki(league1)
        league1_df = merge_datasets(league1_df, league1)
        cities = clean_wiki(league2)
        league2_df = merge_datasets(league2_df, league2)

        # Group W/L by city
        league1_df = league1_df.groupby('metro').agg({'W/L%':np.mean})
        league2_df = league2_df.groupby('metro').agg({'W/L%':np.mean})

        # Merge DFs on metro
        merged = pd.merge(league1_df,league2_df,how = 'inner',on = 'metro')

        # Run t-test for league1-league2
        p_values.loc[league1,league2] = stats.ttest_rel(merged['W/L%_x'],merged['W/L%_y'])[1]
        p_values.loc[league2,league1] = p_values.loc[league1,league2]
        return p_values
    #     return (league1_df,league2_df,cities)


    ## NHL-MLB
    league_comp(nhl_df, 'NHL', 'W/L%', mlb_df, 'MLB', 'W-L%')

    ## NBA-MLB
    league_comp(nba_df, 'NBA', 'W/L%', mlb_df, 'MLB', 'W-L%')

    ## NBA-NHL
    league_comp(league1_df = nba_df, league1 = 'NBA', league1_wl = 'W/L%',
                       league2_df = nhl_df, league2 = 'NHL', league2_wl = 'W/L%')

    ## NFL-MLB
    test = league_comp(nfl_df, 'NFL', 'W-L%', mlb_df, 'MLB', 'W-L%')

    ## NFL-NHL
    league_comp(nfl_df, 'NFL', 'W-L%', nhl_df, 'NHL', 'W/L%')

    ## NFL-NBA
    league_comp(nfl_df, 'NFL', 'W-L%',nba_df, 'NBA', 'W/L%')

    assert abs(p_values.loc["NBA", "NHL"] - 0.02) <= 1e-2, "The NBA-NHL p-value should be around 0.02"
    assert abs(p_values.loc["MLB", "NFL"] - 0.80) <= 1e-2, "The MLB-NFL p-value should be around 0.80"
    return p_values

df = sports_team_performance()
print(
    'Does the success of one league in a city predict the success of another? \nPaired t-test p-values:')
print(df)
