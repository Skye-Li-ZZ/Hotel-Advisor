# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 18:59:41 2021

@author: Skye Li
"""

def data_wrangling(file):
    """

    Parameters
    ----------
    file : csv file

    Returns
    -------
    pd.DataFrame

    """
    
    # import file, and drop duplicates & unwanted column
    import pandas as pd
    df0 = pd.read_csv(file)
    df0 = df0[["region", "name", "star", "rank", "class", "style", "grade_walkers", "n_restaurants", "n_attractions", "n_reviews", "ratings_0", "ratings_1", "ratings_2", "ratings_3", "ratings_4", "reviews_0", "reviews_1", "reviews_2", "reviews_3", "reviews_4"]]
    df0 = df0.drop_duplicates()
    
    # create dummies with "styles"
    #txt = ''
    #for i in range(200):
        #txt = txt + df0['style'].iloc[i]
    #txt_2 = pd.Series(re.findall(pattern="\'[A-Z][A-z 0-9\,]+\'", string= txt))
    #pd.Series(txt_2.unique()).sort_values().to_list()
    categories = ["Bay View", "Boutique", "Budget", "Business", "Centrally Located", "Charming", "City View", "Classic", "English", "Chinese", "French", "Russian", "Spanish", "Arabic", "Dutch", "German", "Italian", "Hungarian", "Portuguese", "Family Resort", "Family", "Great View", "Green", "Harbor View", "Hidden Gem", "Historic Hotel", "Luxury", "Marina View", "Modern", "Ocean View", "Park View", "Quaint", "Quiet", "Quirky Hotels", "Residential Neighborhood", "River View", "Romantic", "Trendy", "Value"]
    dummies    = []
    for k in categories:
        dummies.append(df0['style'].str.contains(k)*1)
    styles_dummy = pd.DataFrame(dummies).transpose()
    styles_dummy.columns = categories
    df0 =  pd.concat([df0, styles_dummy], axis=1)
    
    # pivot table, for reviews and ratings
    tmp = df0[['name', 'reviews_0', 'reviews_1', 'reviews_2', 'reviews_3', 'reviews_4']].copy()
    tmp = tmp.set_index('name')
    df_reviews = pd.DataFrame(tmp.stack(dropna=False)).reset_index()
    df_reviews = df_reviews.set_index('name')
    df_reviews = df_reviews.drop("level_1", axis=1)
    df_reviews.columns = ['reviews']
    
    tmp = df0[['name', 'ratings_0', 'ratings_1', 'ratings_2', 'ratings_3', 'ratings_4']].copy()
    tmp = tmp.set_index('name')
    df_ratings = pd.DataFrame(tmp.stack(dropna=False)).reset_index()
    df_ratings = df_ratings.set_index('name')
    df_ratings = df_ratings.drop("level_1", axis=1)
    df_ratings.columns = ['ratings']
    
    df_ratings_reviews = pd.concat([df_ratings, df_reviews], axis=1)
    df_ratings_reviews = df_ratings_reviews.reset_index()
    df = df0.drop(['ratings_0', 'ratings_1', 'ratings_2', 'ratings_3', 'ratings_4', 'reviews_0', 'reviews_1', 'reviews_2', 'reviews_3', 'reviews_4'], axis=1)
    #df_tmp = df.loc[df.index.repeat(5)].reset_index()
    #df1 = pd.concat([df_tmp, df_ratings_reviews.drop('name', axis=1)], axis=1).drop('index', axis=1)
    df1 = df_ratings_reviews.merge(df.drop_duplicates(subset=['name']),left_on ='name' ,right_on ='name' ,how='left')
    
    # drop rows with null in reviews
    df1 = df1[~df1.reviews.isnull()]
    
    # extract numbers from phrases
    df2 = df1.copy()
    df2['class'] = df2["class"].str.extract("([0-9\.]+)([a-zA-Z0-9]+)")[0]
    df2['n_reviews'] = df2['n_reviews'].str.extract("([0-9]+)( reviews)")[0]
    df2['ratings'] = df2['ratings'].str.extract("(ui_bubble_rating bubble_)([0-9]+)")[1]
    
    # split styles
    #df3 = df2.copy()
    #df3['style'] = df3['style'].str.replace("'", "").str.replace("[", "").str.replace("]", "").str.split(",")
    
    # change data type for numeric columns
    df3 = df2.copy()
    df3[['star', 'class', 'grade_walkers', 'n_restaurants', 'n_attractions', 'n_reviews', 'ratings']] = df3[['star', 'class', 'grade_walkers', 'n_restaurants', 'n_attractions', 'n_reviews', 'ratings']].apply(pd.to_numeric, errors='coerce')
    df3['ratings'] = df3['ratings'] / 10
    
    return df3
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    