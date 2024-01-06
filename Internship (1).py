#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # Rating

# In[2]:


df_rating=pd.read_csv("C:/Users/doppa/Downloads/movie_data_extrt/ratings.csv")
df_rating.head()


# In[3]:


df_rating.shape


# In[4]:


df_rating['userId'].nunique()


# # Movies

# In[5]:


df_movie=pd.read_csv("C:/Users/doppa/Downloads/movie_data_extrt/movies.csv")
df_movie.head()


# # Ratings & Movie

# In[6]:


rating_movie_df = pd.merge(df_rating, df_movie, on='movieId', how='inner')
rating_movie_df


# In[7]:


rating_movie_df.info()


# In[8]:


max_rating = rating_movie_df['rating'].value_counts()
max_rating


# ### What is the average user rating for movie named "Terminator 2: Judgment Day (1991)"?

# In[9]:


terminator_df = rating_movie_df[rating_movie_df['title'] == 'Terminator 2: Judgment Day (1991)']
average_rating = terminator_df['rating'].mean()
print(f"The average user rating for 'Terminator 2: Judgment Day (1991)' is: {average_rating:}")


# # Tags

# In[10]:


# Read the dataset
df_tags = pd.read_csv('C:/Users/doppa/Downloads/movie_data_extrt/tags.csv')
df_tags.head()


# # tags& movie

# In[11]:


tags_movie_df = pd.merge(df_tags, df_movie, on='movieId', how='inner')
tags_movie_df


# ### Which movie has recieved maximum number of user ratings?

# In[12]:


count= rating_movie_df.groupby('movieId')['rating'].count()
max_rating = count.idxmax()

rating_movie_df.loc[rating_movie_df['movieId']==max_rating,'title'].iloc[0]


# ### Select all the correct tags submitted by users to "Matrix, The (1999)" movie?

# In[13]:


tags_movie_rating_df = pd.merge(df_movie,df_tags,on=["movieId"])
tags_movie_rating_df


# In[14]:


tags_movie_rating_df[tags_movie_rating_df['title'] == "Matrix, The (1999)"]


# In[15]:


matrix_movie_id = df_movie.loc[df_movie['title']== "Matrix, The (1999)", 'movieId'].values[0]
matrix_tags = df_tags[df_tags['movieId']==matrix_movie_id]['tag'].unique()
for tag in matrix_tags:
    print(tag)


# ### How does the data distribution of user ratings for "Fight Club (1999)" movie looks like?

# In[16]:


data = rating_movie_df[rating_movie_df["title"]=='Fight Club (1999)']
sns.kdeplot(data["rating"])


# In[17]:


# Read the datasets
movies_df = pd.read_csv('C:/Users/doppa/Downloads/movie_data_extrt/movies.csv')
ratings_df = pd.read_csv('C:/Users/doppa/Downloads/movie_data_extrt/ratings.csv')

grouped_ratings = ratings_df.groupby('movieId').agg({'rating': ['count', 'mean']}).reset_index()
grouped_ratings.columns = ['movieId', 'rating_count', 'rating_mean']

merged_df = pd.merge(movies_df, grouped_ratings, on='movieId', how='inner')
filtered_df = merged_df[merged_df['rating_count'] > 50]

filtered_df


# ### Which movie is the most popular based on  average user ratings?
# 

# In[18]:


most_popular_movie = filtered_df.sort_values(by='rating_mean', ascending=False).head(1)
print("The most popular movie based on average user ratings is:")
print(most_popular_movie[['movieId', 'title', 'rating_count', 'rating_mean']])


# ### Select all the correct options which comes under top 5 popular movies based on number of user ratings.

# In[19]:


top5_popular_movies = filtered_df.sort_values(by='rating_count', ascending=False).head(10)

print("Top 5 popular movies based on number of user ratings:")
print(top5_popular_movies[['movieId', 'title', 'rating_count', 'rating_mean']])


# ### Which Sci-Fi movie is "third most popular" based on the number of user ratings?

# In[20]:


scifi_movies = filtered_df[filtered_df['genres'].str.contains('Sci-Fi', case=False)]

sorted_scifi_movies = scifi_movies.sort_values(by='rating_count', ascending=False)

third_most_popular_scifi_movie = sorted_scifi_movies.iloc[2]

print("The third most popular Sci-Fi movie based on the number of user ratings is:")
print(third_most_popular_scifi_movie[['movieId', 'title', 'rating_count', 'rating_mean']])


# In[21]:


links_df = pd.read_csv('C:/Users/doppa/Downloads/movie_data_extrt/links.csv')
merged_links = pd.merge(filtered_df, links_df, on='movieId', how='inner')
merged_links.head()


# In[31]:


import requests
import numpy as np
from bs4 import BeautifulSoup

def scraper(imdbId):
    id = str(int(imdbId))
    n_zeroes = 7 - len(id)
    new_id = "0" * n_zeroes + id
    URL = f"https://www.imdb.com/title/tt{new_id}/"
    request_header = {
        'Content-Type': 'text/html; charset=UTF-8',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0',
        'Accept-Encoding': 'gzip, deflate, br'
    }
    
    response = requests.get(URL, headers=request_header)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        imdb_rating_tag = soup.find('span', attrs={'data-testid': 'FILL_IN_THE_BLANK'})
        return imdb_rating_tag.text.strip() if imdb_rating_tag else np.nan
    else:
        print(f"Failed to fetch data. Status Code: {response.status_code}")
        return np.nan


# ### Mention the movieId of the movie which has the highest IMDB rating.

# In[32]:


highest_rated_movie = movies_df.loc[movies_df['imdb_rating'].idxmax()]

print("MovieId of the highest-rated movie:", highest_rated_movie['movieId'])


# ### Mention the movieId of the "Sci-Fi" movie which has the highest IMDB rating.

# In[33]:


sci_fi_movies = movies_df[movies_df['genres'].str.contains('Sci-Fi')]

highest_rated_sci_fi_movie = sci_fi_movies.loc[sci_fi_movies['imdb_rating'].idxmax()]

print("MovieId of the highest-rated Sci-Fi movie:", highest_rated_sci_fi_movie['movieId'])


# In[ ]:




