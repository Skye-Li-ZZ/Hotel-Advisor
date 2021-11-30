# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 17:49:17 2021

@author: Skye Li
"""

"""
In this part, I conduct Sentiment Anlysis using 5602 reviews and their associated ratings.
Vectorizer is used to transform plain text into meaningful representation of numbers, so that I can input them into machine learning models.
For model selection, I tried random forest, logistic regression, K nearest neighbor, support vector machine and decision tree. 
At last, word cloud plots were created to show high frequency words for different groups.
"""

#%%%
import pandas                 as pd
import numpy                  as np
import seaborn                as sns
import matplotlib.pyplot      as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit

from sklearn.decomposition           import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from imblearn.over_sampling import SMOTE 
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

from wordcloud import WordCloud, STOPWORDS

import warnings
warnings.filterwarnings('ignore')

#%%%
"""
Part I: Data Preparation
1. Load data from csv file and select only needed columns
2. Clean text of reviews by replacing abbreviations before tokenize
3. Create column "positive" where ratings >= 4.5 (to surpass half of the hotels)
"""
# Load data
df_all = pd.read_csv('all_hotels.csv')
df_sentiment = df_all[["ratings", "reviews"]].copy()

# Clean text of reviews
df_sentiment['reviews'] = df_sentiment['reviews'].str.replace("n't"," not")
df_sentiment['reviews'] = df_sentiment['reviews'].str.replace("I'm","I am")
df_sentiment['reviews'] = df_sentiment['reviews'].str.replace("'ll"," will")
df_sentiment['reviews'] = df_sentiment['reviews'].str.replace("It's","It is")
df_sentiment['reviews'] = df_sentiment['reviews'].str.replace("it's","It is")
df_sentiment['reviews'] = df_sentiment['reviews'].str.replace("that's","that is")

# Create binary variable
positive = pd.DataFrame([df_sentiment['ratings']>=4.5]).transpose()
positive.columns = ["positive"]
df_sentiment = pd.concat([df_sentiment, positive], axis=1)
df_sentiment["positive"] = df_sentiment["positive"]*1


#%%%
"""
Part II: Text transformation
1. Transform plain text to vector of numbers
2. Decrease dimensionality of vector, using SVD
"""
# Create a corpus, stop_words added
corpus = df_sentiment["reviews"].to_list()
vectorizer_count = CountVectorizer(lowercase = True,ngram_range = (2,3), max_df = 0.96, min_df = 0.001, stop_words="english")
X = vectorizer_count.fit_transform(corpus)

## This plot shows the frequency of words
features_frequency   = pd.DataFrame({'feature' : vectorizer_count.get_feature_names(),'feature_frequency' : X.toarray().sum(axis=0)})
# X.shape #(5602, 4704)
sns.barplot(x="feature", y="feature_frequency", data=features_frequency.sort_values(by='feature_frequency',ascending=False).head(50))
plt.xticks(rotation=90)

# Dimension reduction
# Keep 80% of information with about 1500 components
#svd = TruncatedSVD(n_components=(X.shape[1]-1), n_iter=7, random_state=42)
svd = TruncatedSVD(n_components=(1500), n_iter=7, random_state=42)
svd.fit(X)
X = svd.transform(X)
pd.Series(svd.explained_variance_ratio_).plot()
pd.Series(np.cumsum(svd.explained_variance_ratio_)).plot()


#%%%
"""
Part III: Machine Learning
1. Perform train/test split
2. Model Training
    a. Using grid search to choose hyperparameters
        For each grid search:
            a) Create the base model
            b) Create the parameter grid
            c) Implement search
            d) Find best parameters and train the model
            e) Test model on test set
3. Comparison between models, considering accuracy, ROC, AUC   
"""

"""
1. Perform train/test split
"""
X = pd.DataFrame(X)
y = df_sentiment["positive"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

#%%%
"""
2. Model Training
"""
# Prepare for grid search, set up cross-validation shuffles
shuffle_split = ShuffleSplit(n_splits = 3, test_size=0.1)


""" Random Forest""" 

rf = RandomForestClassifier(max_features=5,random_state=42)
param_rf = {'max_depth': [50, 100, 500], 'n_estimators': [10, 100, 500]}
grid_search_rf = GridSearchCV(estimator = rf, param_grid = param_rf, cv = shuffle_split, return_train_score=True, n_jobs = -1)

grid_search_rf.fit(X_train,y_train)
results_rf = pd.DataFrame(grid_search_rf.cv_results_)

print(grid_search_rf.best_params_)
print(grid_search_rf.best_estimator_)
best_model_rf = grid_search_rf.best_estimator_
best_model_rf.fit(X_train,y_train)
print("Tests score with best parameters: ", best_model_rf.score(X_test,y_test))

y_pred_rf = best_model_rf.predict(X_test)
fpr_rf, tpr_rf, thresholds = roc_curve(y_test, y_pred_rf)

#%%%
""" Logistic Regression""" 

lg_clf = LogisticRegression(solver='lbfgs', max_iter=5000,random_state=42)
param_lg = {'penalty': ['l1', 'l2'], 'C': [1e-2, 1e-1, 1, 10, 100]}
grid_search_lg = GridSearchCV(estimator = lg_clf, param_grid = param_lg, cv = shuffle_split, return_train_score=True, n_jobs = -1)

grid_search_lg.fit(X_train, y_train)
results_lg = pd.DataFrame(grid_search_lg.cv_results_)

print(grid_search_lg.best_params_)
print(grid_search_lg.best_estimator_)
best_model_lg = grid_search_lg.best_estimator_
best_model_lg.fit(X_train, y_train)
print("Tests score with best parameters: ", best_model_lg.score(X_test,y_test))

y_pred_lg = best_model_lg.predict(X_test)
fpr_lg, tpr_lg, thresholds = roc_curve(y_test, y_pred_lg)

#%%%
""" K Nearest Neighbor""" 

knn = KNeighborsClassifier()
param_knn = {'n_neighbors': np.arange(2,20,1), 'weights': ('uniform', 'distance')}
grid_search_knn = GridSearchCV(estimator = knn, param_grid = param_knn, cv = shuffle_split, return_train_score=True, n_jobs = -1)

grid_search_knn.fit(X_train,y_train)
results_knn = pd.DataFrame(grid_search_knn.cv_results_)

print(grid_search_knn.best_params_)
print(grid_search_knn.best_estimator_)
best_model_knn = grid_search_knn.best_estimator_
best_model_knn.fit(X_train,y_train)
print("Tests score with best parameters: ", best_model_knn.score(X_test,y_test))

y_pred_knn = best_model_knn.predict(X_test)
fpr_knn, tpr_knn, thresholds = roc_curve(y_test, y_pred_knn)

#%%%
""" Support Vector Machine"""

svm = SVC(random_state=42)
param_svc = {'kernel': ['linear', 'poly', 'rbf'], 'C': [0, 0.1, 1.], 'gamma': [0.1, 1.]}
grid_search_svm = GridSearchCV(estimator = svm, param_grid = param_svc, cv = shuffle_split, return_train_score=True, n_jobs = -1)

grid_search_svm.fit(X_train,y_train)
results_svm = pd.DataFrame(grid_search_svm.cv_results_)

print(grid_search_svm.best_params_)
print(grid_search_svm.best_estimator_)
best_model_svm = grid_search_svm.best_estimator_
best_model_svm.fit(X_train,y_train)
print("Tests score with best parameters: ", best_model_svm.score(X_test,y_test))

y_pred_svm = best_model_svm.predict(X_test)
fpr_svm, tpr_svm, thresholds = roc_curve(y_test, y_pred_svm)

#%%%
""" Decision Tree""" 

dtc = DecisionTreeClassifier(random_state=42)
param_dtc = {'max_depth': [5, 10, 100, 500], 'max_features': [5, 50, 100]}
grid_search_dtc = GridSearchCV(estimator = dtc, param_grid = param_dtc, cv = shuffle_split, return_train_score=True, n_jobs = -1)

grid_search_dtc.fit(X_train,y_train)
results_dtc = pd.DataFrame(grid_search_dtc.cv_results_)

print(grid_search_dtc.best_params_)
print(grid_search_dtc.best_estimator_)
best_model_dtc = grid_search_dtc.best_estimator_
best_model_dtc.fit(X_train,y_train)
print("Tests score with best parameters: ", best_model_dtc.score(X_test,y_test))

y_pred_dtc = best_model_dtc.predict(X_test)
fpr_dtc, tpr_dtc, thresholds = roc_curve(y_test, y_pred_dtc)


#%%%
"""
4. Model Comparison
"""

# Simple ROC curve plot
plt.plot(fpr_rf,tpr_rf,label='rf')
plt.plot(fpr_lg,tpr_lg,label='lg')
plt.plot(fpr_knn,tpr_knn,label='knn')
plt.plot(fpr_svm,tpr_svm,label='svm')
plt.plot(fpr_dtc,tpr_dtc,label='dtc')
plt.xlabel('False positive rate')
plt.ylabel('Recall: True negative rate')
plt.legend()
plt.grid()

# Calculate AUC score
y_probs_rf  = best_model_rf.predict_proba(X_test)[:,1]
y_probs_lg  = best_model_lg.predict_proba(X_test)[:,1]
y_probs_knn  = best_model_knn.predict_proba(X_test)[:,1]
#y_probs_svm  = best_model_svm.predict_proba(X_test)[:,1]
y_probs_dtc  = best_model_dtc.predict_proba(X_test)[:,1]
print("AUC (rf):",roc_auc_score(y_test, y_probs_rf))
print("AUC (lg):",roc_auc_score(y_test, y_probs_lg))
print("AUC (knn):",roc_auc_score(y_test, y_probs_knn))
#print("AUC (svm):",roc_auc_score(y_test, y_probs_svm))
print("AUC (dtc):",roc_auc_score(y_test, y_probs_dtc))
#%%%
"""
Part IV: Visualization
1. Find key words for positive / non-positive ratings
2. Interpretation
"""
stopwords = set(STOPWORDS)
stopwords.update(["room", "stay", "stayed", "room", "rooms", "hotel", "place", "one"])

good_reviews = df_sentiment[df_sentiment["positive"]==1].reviews.to_list()
positive = " ".join(good_reviews)
testcloud = WordCloud(stopwords=stopwords, background_color='white',).generate(positive)
plt.imshow(testcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

bad_reviews = df_sentiment[df_sentiment["positive"]==0].reviews.to_list()
negative = " ".join(good_reviews)
testcloud = WordCloud(stopwords=stopwords, background_color='white',).generate(negative)
plt.imshow(testcloud, interpolation='bilinear')
plt.axis("off")
plt.show()



