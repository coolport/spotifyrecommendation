import pandas as pd
import numpy as np
import json
import re 
import sys
import itertools

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt





def preprocess(spotify_df, data_w_genre):
    
    #AT THIS POINT, spotify_df, data_w_genre(filtered already) has been read;
    data_w_genre=data_w_genre.drop('id', axis=1) #dropped ID column here
    
    #PRE PROCESS PART
    spotify_df['artists_upd_v1'] = spotify_df['artists'].apply(lambda x: re.findall(r"'([^']*)'", x))

    #strings = spotify_df['artists'].values[0] #uli

    spotify_df['artists_upd_v2'] = spotify_df['artists'].apply(lambda x: re.findall('\"(.*?)\"',x))
    spotify_df['artists_upd'] = np.where(spotify_df['artists_upd_v1'].apply(lambda x: not x), spotify_df['artists_upd_v2'], spotify_df['artists_upd_v1'] )

    spotify_df['artists_upd'] = spotify_df['artists_upd'].apply(lambda x: x if isinstance(x, list) and x else [''])
    spotify_df['name'] = spotify_df['name'].fillna('')
    #this removes null value, that's all



    spotify_df['artists_song'] = spotify_df.apply(lambda row: row['artists_upd'][0]+row['name'],axis = 1)
    # Create a unique song identifier

    spotify_df.sort_values(['artists_song','release_date'], ascending = False, inplace = True)

    spotify_df.drop_duplicates('artists_song',inplace = True)

    #EXPLODE#EXPLODE#EXPLODE
    #EXPLODE#EXPLODE#EXPLODE
    #EXPLODE#EXPLODE#EXPLODE

    artists_exploded = spotify_df[['artists_upd','id']].explode('artists_upd')

    artists_exploded_enriched = artists_exploded.merge(data_w_genre, how = 'left', left_on = 'artists_upd',right_on = 'name')
    artists_exploded_enriched_nonnull = artists_exploded_enriched[~artists_exploded_enriched.genres_upd.isnull()]


    artists_genres_consolidated = artists_exploded_enriched_nonnull.groupby('id')['genres_upd'].apply(list).reset_index()

    artists_genres_consolidated['consolidates_genre_lists'] = artists_genres_consolidated['genres_upd'].apply(lambda x: list(set(list(itertools.chain.from_iterable(x)))))

    spotify_df = spotify_df.merge(artists_genres_consolidated[['id','consolidates_genre_lists']], on = 'id',how = 'left')

    #FEATURE ENGINEERING
    #FEATURE ENGINEERING
    #FEATURE ENGINEERING
    #FEATURE ENGINEERING


    spotify_df['year'] = spotify_df['release_date'].apply(lambda x: x.split('-')[0])
    float_cols = spotify_df.dtypes[spotify_df.dtypes == 'float64'].index.values
    ohe_cols = 'popularity'
    spotify_df['popularity_red'] = spotify_df['popularity'].apply(lambda x: int(x/5))

    # tfidf can't handle nulls so fill any null values with an empty list
    spotify_df['consolidates_genre_lists'] = spotify_df['consolidates_genre_lists'].apply(lambda d: d if isinstance(d, list) else [])

    return spotify_df, float_cols





