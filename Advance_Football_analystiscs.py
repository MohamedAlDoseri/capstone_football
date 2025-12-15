#!/usr/bin/env python
# coding: utf-8

# In[38]:





# In[39]:


#load all the csv files
import pandas as pd
# load all the csv files


appearances = pd.read_csv(r"C:\Users\John\Downloads\appearances up.csv")
clubs = pd.read_csv(r"C:\Users\John\Downloads\clubs.csv")
competitions = pd.read_csv(r"C:\Users\John\Downloads\competitions.csv")
player_valuations = pd.read_csv(r"C:\Users\John\Downloads\player_valuations.csv")
players = pd.read_csv(r"C:\Users\John\Downloads\players New.csv")
transfers = pd.read_csv(r"C:\Users\John\Downloads\transfers New.csv")


# In[40]:


# first 5 rows of apperance csv
appearances.head()


# In[41]:


#show 5 first 5 rows of player valuationcsv
player_valuations.head()


# In[42]:


#merge appearance csv with player csv using player id
app_players = appearances.merge(players, on='player_id', how='left')


# In[43]:


#on app_players create a new column named "goal_contribution" = goals+assists
app_players['goal_contribution'] = app_players['goals'] + app_players['assists']


# In[44]:


#test relationship btn player valuation and goal contribution
app_players['goal_contribution'].corr(app_players['market_value_in_eur'])


# In[45]:


import os
os.listdir('/content')


# In[46]:


#show club csv head
clubs.head()


# In[47]:


app_players.head()


# In[48]:


import matplotlib.pyplot as plt

# --- Load data ---
appearances = pd.read_csv(r"C:\Users\John\Downloads\appearances up.csv")
players = pd.read_csv(r"C:\Users\John\Downloads\players New.csv")
# --- Compute goal contributions ---
appearances['goal_contribution'] = appearances['goals'] + appearances['assists']

# --- Merge ---
app_players = appearances.merge(players, on='player_id', how='left')

# --- Filter out zeros to avoid division crashes ---
app_players = app_players[(app_players['market_value_in_eur'] > 0) &
                          (app_players['goal_contribution'] > 0)]

# --- Compute ratio ---
app_players['value_per_goal_contribution'] = (
    app_players['market_value_in_eur'] / app_players['goal_contribution']
)

# --- Top 10 ---
top10 = app_players.sort_values(
    by='value_per_goal_contribution', ascending=False
).head(10)

# --- Plot ---
plt.figure(figsize=(12, 7))
bars = plt.barh(
    top10['player_name'],
    top10['value_per_goal_contribution'],
)

# Color each bar by club
clubs = top10['current_club_name'].fillna("Unknown")
colors = plt.cm.viridis(top10['player_current_club_id'].rank(method='dense')/len(top10['player_current_club_id'].unique()))

for bar, color, club in zip(bars, colors, clubs):
    bar.set_color(color)

# Add a legend
handles = [plt.Rectangle((0,0),1,1, color=plt.cm.viridis(i)) for i in top10['player_current_club_id'].rank(method='dense').sort_values().unique()/len(top10['player_current_club_id'].unique())]
plt.legend(handles, top10.sort_values(by='player_current_club_id')['current_club_name'].unique().tolist(), title="Club", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.xlabel('Value per Goal Contribution (€/GC)')
plt.ylabel('Player Name')
plt.title('Top 10 Players by Market Value per Goal Contribution')
plt.tight_layout()
plt.show()


# Merge datasets players new and apperance
# Then we run the regression

# In[49]:


import statsmodels.api as sm
import numpy as np

# --- Compute goal contributions ---
appearances['goal_contribution'] = appearances['goals'] + appearances['assists']

# --- Merge appearances with players ---
df = appearances.merge(players, on='player_id', how='left')

# --- Select regression variables ---
reg_data = df[[
    'market_value_in_eur', 'goals', 'assists', 'minutes_played',
    'height_in_cm', 'date_of_birth'
]].copy()

# --- Drop missing values ---
reg_data = reg_data.dropna()

# --- Convert DOB to age ---
reg_data['date_of_birth'] = pd.to_datetime(reg_data['date_of_birth'], errors='coerce')
reg_data = reg_data.dropna(subset=['date_of_birth'])   # ensure valid dates

reg_data['age'] = (
    pd.to_datetime("2022-01-01") - reg_data['date_of_birth']
).dt.days / 365.25

# --- Remove impossible ages ---
reg_data = reg_data[(reg_data['age'] > 15) & (reg_data['age'] < 45)]

# --- Define X and y ---
y = reg_data['market_value_in_eur']
X = reg_data[['goals', 'assists', 'minutes_played', 'age', 'height_in_cm']]

# --- Add constant for OLS ---
X = sm.add_constant(X)

# --- Fit OLS model ---
model = sm.OLS(y, X).fit()

print(model.summary())


# In[50]:


#save regressionn results into csv



# In[50]:





# In[51]:


import matplotlib.pyplot as plt
import pandas as pd

# Make sure values are numeric
app_players['market_value_in_eur'] = pd.to_numeric(app_players['market_value_in_eur'], errors='coerce')
app_players['goal_contribution'] = pd.to_numeric(app_players['goal_contribution'], errors='coerce')

# Drop missing values
plot_df = app_players.dropna(subset=['market_value_in_eur', 'goal_contribution'])

# Optional: remove extreme outliers (top 1%) to improve visibility
plot_df = plot_df[
    plot_df['market_value_in_eur'] < plot_df['market_value_in_eur'].quantile(0.99)
]

plt.figure(figsize=(10,6))
plt.scatter(
    plot_df['goal_contribution'],
    plot_df['market_value_in_eur'],
    alpha=0.3,
    s=10
)

plt.xlabel("Goal Contributions (Goals + Assists)")
plt.ylabel("Market Value (€)")
plt.title("Player Market Value vs Goal Contributions")
plt.grid(True)
plt.show()


# In[52]:


#add column


# In[53]:


import pandas as pd

df = app_players.copy()

# Ensure numeric
df['market_value_in_eur'] = pd.to_numeric(df['market_value_in_eur'], errors='coerce')
df['goal_contribution'] = pd.to_numeric(df['goal_contribution'], errors='coerce')

# Remove missing or zero goal contributions
df = df.dropna(subset=['market_value_in_eur', 'goal_contribution'])
df = df[df['goal_contribution'] > 0]

# Compute ratio
df['value_to_gc_ratio'] = df['market_value_in_eur'] / df['goal_contribution']

# Aggregate by player (some players appear multiple times)
player_ranking = df.groupby(['player_id', 'player_name']).agg({
    'market_value_in_eur': 'max',
    'goal_contribution': 'sum',
    'value_to_gc_ratio': 'mean'
}).reset_index()

# Sort descending
top10 = player_ranking.sort_values(by='value_to_gc_ratio', ascending=False).head(10)

top10


# In[54]:


from matplotlib import pyplot as plt
top10['player_id'].plot(kind='hist', bins=20, title='player_id')
plt.gca().spines[['top', 'right',]].set_visible(False)


# In[55]:


#plot player goal contributions vs valuation
import matplotlib.pyplot as plt
plt.scatter(app_players['goal_contribution'], app_players['market_value_in_eur'])


# In[56]:


#clean app_players and drop nana
app_players = app_players.dropna()


# In[57]:


#rrgression line for goal contribution vas valuation
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(app_players[['goal_contribution']], app_players['market_value_in_eur'])


# In[58]:


#plot the regression line for the relationship
plt.scatter(app_players['goal_contribution'], app_players['market_value_in_eur'])
plt.plot(app_players['goal_contribution'], model.predict(app_players[['goal_contribution']]), color='red')
#


# In[59]:


#save the cleaned app_players to a csv file
app_players.to_csv('app_players_cleaned.csv', index=False)


# In[60]:


#we plot a bar graph of top 10 valuation players
top_players = app_players.sort_values(by='market_value_in_eur', ascending=False).head(10)
plt.bar(top_players['name'], top_players['market_value_in_eur'])


# In[61]:


#plot Age vs Performance Curve
import matplotlib.pyplot as plt

# Calculate average market value per club from app_players
avg_market_value_per_club = app_players.groupby('player_current_club_id')['market_value_in_eur'].mean().reset_index()

# Merge with clubs dataframe on club_id
club_age_value = clubs.merge(avg_market_value_per_club, left_on='club_id', right_on='player_current_club_id', how='inner')

# Plot the scatter plot
plt.scatter(club_age_value['average_age'], club_age_value['market_value_in_eur'])
plt.xlabel('Average Age of Club')
plt.ylabel('Average Market Value of Players (EUR)')
plt.title('Average Club Age vs. Average Player Market Value')
plt.show()


# In[62]:


#player valuation by position bar graph
position_market_value = app_players.groupby('position')['market_value_in_eur'].mean().reset_index()
plt.bar(position_market_value['position'], position_market_value['market_value_in_eur'])


# In[63]:


#save app_players to csv
app_players.to_csv('app_players.csv', index=False)


# In[64]:


import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Ensure numeric
app_players['goals'] = pd.to_numeric(app_players['goals'], errors='coerce').fillna(0)
app_players['assists'] = pd.to_numeric(app_players['assists'], errors='coerce').fillna(0)
app_players['minutes_played'] = pd.to_numeric(app_players['minutes_played'], errors='coerce').fillna(0)

# Compute GC and GC/90
app_players['goal_contribution'] = app_players['goals'] + app_players['assists']
app_players['gc90'] = app_players['goal_contribution'] / (app_players['minutes_played'] / 90)

# Remove invalid values
app_players = app_players.replace([np.inf, -np.inf], np.nan)
app_players = app_players.dropna(subset=['gc90', 'position'])

# ANOVA model
model = ols('gc90 ~ C(position)', data=app_players).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

anova_table


# In[ ]:





# In[65]:


#from clubs, rank the top 10 market value
top_clubs = clubs.sort_values(by='total_market_value', ascending=False).head(10)
plt.bar(top_clubs['name'], top_clubs['total_market_value'])


# In[66]:


#show first 5 rows of clubs csv
clubs.head()


# In[67]:


#show head for transfer
transfers.head()


# In[ ]:


#drop ns club and save as clean csv
clubs = clubs.dropna()
clubs.to_csv('clubs_cleaned.csv', index=False)


# In[ ]:


#highest transfer value
transfers.sort_values(by='transfer_fee', ascending=False).head()


# In[ ]:


#merge transfer csv with apperance csv and name it app_transfer and save as csv
app_transfer = app_players.merge(transfers, on='player_id', how='left')
app_transfer.to_csv('app_transfer.csv', index=False)


# In[ ]:


#download the app_transfer csv
from google.colab import files
files.download('app_transfer.csv')


# In[ ]:


#Show first 5rows of club csv
clubs.head()


# In[ ]:


#show first 5rows of transfers
transfers.head()


# Convert Datatypes

# In[ ]:


# Convert club IDs to Int (nullable)
appearances['player_club_id'] = appearances['player_club_id'].astype('Int64')
appearances['player_current_club_id'] = appearances['player_current_club_id'].astype('Int64')

# Fix date type
appearances['date'] = pd.to_datetime(appearances['date'], errors='coerce')

player_valuations['date'] = pd.to_datetime(player_valuations['date'], errors='coerce')
transfers['transfer_date'] = pd.to_datetime(transfers['transfer_date'], errors='coerce')
players['date_of_birth'] = pd.to_datetime(players['date_of_birth'], errors='coerce')


# Merge Appearances and players

# In[ ]:


app_players = appearances.merge(
    players,
    on='player_id',
    how='left',
    suffixes=('', '_player')
)


# MERGE COMPETITION

# In[ ]:


app_players_comp = app_players.merge(
    competitions[['competition_id', 'name', 'type', 'country_name']],
    on='competition_id',
    how='left',
    suffixes=('', '_competition')
)


# 4. MERGE CLUBS TWICE
# 
# We want:
# 
# the club the player appeared for
# 
# the player’s current club (that season)

# In[ ]:


# Merge player_club_id → club player was representing in that match
app_with_club = app_players_comp.merge(
    clubs.add_prefix('club_'),
    left_on='player_club_id',
    right_on='club_club_id',
    how='left'
)

# Merge player_current_club_id → current club metadata
app_with_both_clubs = app_with_club.merge(
    clubs.add_prefix('currentclub_'),
    left_on='player_current_club_id',
    right_on='currentclub_club_id',
    how='left'
)


# In[ ]:


player_appearances_extended = app_with_both_clubs
player_appearances_extended.shape


#         We BUILD VALUATIONS MASTER TABLE

# In[ ]:


player_valuations_extended = player_valuations.merge(
    players[['player_id', 'name', 'position', 'country_of_citizenship']],
    on='player_id',
    how='left'
).merge(
    clubs.add_prefix('club_'),
    left_on='current_club_id',
    right_on='club_club_id',
    how='left'
)


# TRANSFERS MASTER TABLE

# In[ ]:


player_transfers_extended = transfers.merge(
    players[['player_id', 'name', 'position', 'date_of_birth']],
    on='player_id',
    how='left'
).merge(
    clubs.add_prefix('fromclub_'),
    left_on='from_club_id',
    right_on='fromclub_club_id',
    how='left'
).merge(
    clubs.add_prefix('toclub_'),
    left_on='to_club_id',
    right_on='toclub_club_id',
    how='left'
)


# FULL PLAYER CAREER MASTER TABLE

# In[ ]:


full_player_career = player_appearances_extended.merge(
    player_valuations[['player_id', 'date', 'market_value_in_eur']],
    on=['player_id'],
    how='left'
).merge(
    transfers[['player_id', 'transfer_date', 'from_club_id', 'to_club_id']],
    on='player_id',
    how='left'
)

