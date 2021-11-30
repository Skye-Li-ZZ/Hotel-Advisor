# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 18:09:19 2021

@author: Skye Li
"""

"""
The purpose of this part is to predict how good your hotel is based on information like surrounding evironment (how many resaturants within 3 miles? etc.), standard class level, and other related services.
To be better than about half of the hotels, you need a star greater than 4.5. Thus, let's set it as threshold to seperate "good" and "bad" hotels.
"""

#%%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

from sklearn.inspection import PartialDependenceDisplay
from lime import lime_tabular

#%%%
"""
Part I: Import Data and Extract Target & Features
1. Load data from csv file and select needed columns
2. Create target variable to seperate hotels
"""

""" 1. Load data from csv file and select needed columns """
df_all = pd.read_csv("all_hotels.csv")

# select needed columns
df_star = df_star = df_all[["region", "name", "star", "class", "grade_walkers", "n_restaurants", "n_attractions", "n_reviews", "ratings",
"Bay View", "Boutique", "Budget", "Business", "Centrally Located",  "Charming", "City View", "Classic", "English", "Chinese", "French", "Russian", "Spanish", "Arabic", "Dutch", "German", "Italian", "Hungarian", "Portuguese", "Family Resort", "Family", "Great View", "Green", "Harbor View", "Hidden Gem", "Historic Hotel", "Luxury", "Marina View", "Modern", "Ocean View", "Park View", "Quaint", "Quiet", "Quirky Hotels", "Residential Neighborhood", "River View", "Romantic", "Trendy", "Value"]].copy()

# Create dummy variables for "region"
df_region = pd.get_dummies(df_star[["region"]],drop_first=True)
df_star = pd.concat([df_star, df_region], axis=1)

# drop rows with missing values
df_star = df_star.dropna()
df_star = df_star.reset_index()

# basic summary statistics, we have 2522 instances of records
df_star.info()
df_star.describe()

""" 2. Create target variable to seperate hotels """
# To surpass half of the hotels, you need a star greater than 4.5
np.mean(df_star.star >= 4.5)

# target variable y
df_star['good'] = df_star.star >= 4.5
df_star['good'] = df_star['good'] * 1
y = df_star.good

# features X
X = df_star[["class", "grade_walkers", "n_restaurants", "n_attractions", "n_reviews", "ratings","Bay View", "Boutique", "Budget", "Business", "Centrally Located", "Charming", "City View", "Classic", "English", "Chinese", "French", "Russian", "Spanish", "Arabic", "Dutch", "German", "Italian", "Hungarian", "Portuguese", "Family Resort", "Family", "Great View", "Green", "Harbor View", "Hidden Gem", "Historic Hotel", "Luxury", "Marina View", "Modern", "Ocean View", "Park View", "Quaint", "Quiet", "Quirky Hotels", "Residential Neighborhood", "River View", "Romantic", "Trendy", "Value", 'region_Bristol', 'region_CapeCod', 'region_GreaterBoston','region_GreaterMerrimackValley', 'region_GreaterSpringfield','region_Hampshire', 'region_MarthasVineyard','region_Nantucket', 'region_NorthBoston','region_Plymouth']].copy()

#%%%
"""
Part II: Machine Learning
1. Conduct Train /  Test split
2. Perform grid search with cross-validation to build and fine-tuning Random Forest model
3. Test model using test set
"""
""" 1. Conduct Train /  Test split """
# 90% training set, 10% test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

""" 2. Perform cross validation to build and find-tuning Random Forest model """
# Prepare for grid search, set up cross-validation shuffles
shuffle_split = ShuffleSplit(n_splits = 50, test_size=0.1)

# Set up base model
rf = RandomForestClassifier(max_features=10,random_state=42)

# Set up grid search
param_rf = {'max_depth': [50, 100, 500], 'n_estimators': [10, 100, 500]}
grid_search_rf = GridSearchCV(estimator = rf, param_grid = param_rf, cv = shuffle_split, return_train_score=True, n_jobs = -1)

# Perform grid search and store best parameters
grid_search_rf.fit(X_train,y_train)
results_rf = pd.DataFrame(grid_search_rf.cv_results_)
print(grid_search_rf.best_params_)
print(grid_search_rf.best_estimator_)

# Build model based on best parameters and train it with the whole training set
best_model_rf = grid_search_rf.best_estimator_
best_model_rf.fit(X_train,y_train)

""" 3. Test model using test set """
# Accuracy score of 0.99
print("Tests score with best parameters: ", best_model_rf.score(X_test,y_test))

# Confusion matrix
y_pred = best_model_rf.predict(X_test)
confusion_matrix(y_test, y_pred)

# Precision, Recall, F1-score
Precision = precision_score(y_test, y_pred)
Recall = recall_score(y_test, y_pred)
F1_score = f1_score(y_test, y_pred)

# ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.plot(fpr,tpr)
plt.xlabel('False positive rate')
plt.ylabel('Recall: True negative rate')
plt.legend()
plt.grid()

y_probs  = best_model_rf.predict_proba(X_test)[:,1]
print("AUC:",roc_auc_score(y_test, y_probs))

## This model is good enough!

#%%%
"""
Part III: Model Interpretation
1. Feature importance plot
2. Partial dependence of features
3. Case study for individual hotel, using LIME
"""

""" 1. Feature importance plot """
# Plot the top 20 important features
feat_importances = pd.Series(best_model_rf.feature_importances_, index=X.columns)
feat_importances.nlargest(20).plot(kind='barh')


""" 2. Partial dependence of top 6 features """
features = feat_importances.nlargest(6).index.to_list()
# First 3 features
PartialDependenceDisplay.from_estimator(best_model_rf, X, features[:3],kind='average')
# Top4 to 6 features
PartialDependenceDisplay.from_estimator(best_model_rf, X, features[3:],kind='average')


""" 3. Case study for individual hotel, using LIME """
lime_explainer = lime_tabular.LimeTabularExplainer(training_data=np.array(X), feature_names=X.columns, class_names=['bad', 'good'], mode='classification')

# 5 Star Hotel: Brewster By The Sea Inn
lime_exp = lime_explainer.explain_instance(data_row=X.iloc[213], predict_fn=best_model_rf.predict_proba)
lime_exp.as_pyplot_figure()

# 2.5 Star Hotel: All Seasons Inn And Suites
lime_exp = lime_explainer.explain_instance(data_row=X.iloc[2492], predict_fn=best_model_rf.predict_proba)
lime_exp.as_pyplot_figure()


