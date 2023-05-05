# %% [markdown]
# # MDLE - Collaborative Filtering Assignment
# ### Exercise 2
# ##### Authors: Pedro Duarte 97673, Pedro Monteiro 97484

# %% [markdown]
# Import necessary modules

# %%
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

import pandas as pd
import numpy as np

from operator import add
import math

# %% [markdown]
# Declare constants

# %%
# Spark Constants
APP_NAME = 'assignment1'
MASTER = 'local[*]'

#random hyper planes
# Data Columns
COLUMN_USER_ID = 'userId'
COLUMN_MOVIE_ID = 'movieId'
COLUMN_RATING = 'rating'
COLUMN_TIMESTAMP = 'timestamp'

# Input Constants
MOVIES_INPUT_FILE = 'movies.csv'
TAGS_INPUT_FILE = 'tags.csv'
RATINGS_INPUT_FILE = 'ratings.csv'

# %% [markdown]
# Configuration and Initialization of Spark
# 
# - Parameters:
#     - `APP_NAME` (string): the name of the Spark application
#     - `MASTER` (string): the URL of the Spark master node
# <br></br>
# - Returns:
#     - `sc` (SparkContext): the Spark context for the given application and master
#     - `spark` (SparkSession): the Spark session for the given application and master

# %%
conf = SparkConf().setAppName(APP_NAME).setMaster(MASTER)
sc = SparkContext.getOrCreate(conf=conf)

spark = SparkSession.builder.appName(APP_NAME).master(MASTER).getOrCreate()

# %% [markdown]
# Read ratings CSV file

# %%
ratings = pd.read_csv('ratings.csv')

# %% [markdown]
# Create a function to return a matrix of items (movies) with user ratings

# %%
def create_item_matrix(dataset):
    return dataset.pivot(index=COLUMN_MOVIE_ID, columns=COLUMN_USER_ID, values=COLUMN_RATING).fillna(0)
                                                                            # missing values are filled with 0

# %% [markdown]
# Create a function that clusters the ratings in a given dataset using Agglomerative Clustering algorithm

# %%
def cluster_ratings(dataset):
    X = dataset.values # get dataset values
    X = StandardScaler().fit(X).transform(X) # standardize X

    return AgglomerativeClustering(n_clusters=None, distance_threshold=200).fit_predict(X) # apply Agglomerative Clustering

# %% [markdown]
# Predict the rating that a given user would give to a given movie, based on the user's ratings of similar movies and the similarity of those movies to the given movie, using a collaborative filtering approach

# %%
def predict_rating(item_matrix, clusters, user_id, movie_id):
    if movie_id not in item_matrix.index.to_list(): return 0 # check if the movie_id is present in the item_matrix

    movie_idx = item_matrix.index.get_loc(movie_id) # get movie index
    movie_to_predict = item_matrix.loc[movie_id] # retrieve movie_id row

    similar_movies = item_matrix[clusters == clusters[movie_idx]] #  select all movies in the same cluster as the movie_id

    # check if user_id is present in the similar_movies column
    if user_id not in similar_movies.columns.to_list(): return movie_to_predict.mean() 

    # remove any movies that the user has not rated
    similar_movies = similar_movies[similar_movies[user_id] != 0]

    # calculate cosine similarity between movie_to_predict and all the movies in similar_movies
    distances = np.dot(movie_to_predict, similar_movies.T)/(np.linalg.norm(movie_to_predict)*np.linalg.norm(similar_movies.T))
    
    user_ratings = similar_movies.get(user_id) # get user ratings

    if user_ratings is None: return movie_to_predict.mean() # check if user_ratings is None => return the mean rating

    # calculate weighted sum of the user ratings for the similar movies using the similarity scores as weights
    ratings_product = np.dot(user_ratings, distances).sum()
    user_ratings_distance_total = distances.sum()

    if user_ratings_distance_total == 0: return movie_to_predict.mean()

    # calculate final rating prediction by dividing the weighted sum of the ratings by the sum of the similarity scores
    rating_prediction = ratings_product/user_ratings_distance_total

    return rating_prediction

# %% [markdown]
# Generate a rating prediction for user 416 and movie 319

# %%
ds = create_item_matrix(ratings)
clusters = cluster_ratings(ds)

predict_rating(ds, clusters, 416, 319)

# %% [markdown]
# # 2.1

# %% [markdown]
# Split ratings dataset into training and testing 

# %%
ratings_count = ratings.shape[0] # number of rows

ratings_test_count = math.ceil(ratings_count*.1) # number of ratings that will be used for testing

# shuffle the ratings rows dataset randomly
ratings = ratings.sample(frac = 1)

# set 10% of ratings for testing
ratings_test = ratings[:ratings_test_count]
ratings_train = ratings[ratings_test_count:]


# %% [markdown]
# Generate an item matrix and apply clustering to the training data subset

# %%
train_ds = create_item_matrix(ratings_train)
train_clusters = cluster_ratings(train_ds)

# %% [markdown]
# Calculate the deviation between predicted and actual ratings on the testing subset 

# %%
deviations = sc.parallelize(ratings_test.values) \
  .map(lambda v: abs(predict_rating(train_ds, train_clusters, v[0], v[1]) - v[2])) \
  .reduce(add)

print("Avg Deviation:", deviations/ratings_test_count)


