#!/usr/bin/env python
# coding: utf-8

# In[197]:


from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')


# In[198]:


data = pd.read_csv('train.csv', index_col='id')


# In[199]:


data.count()


# In[200]:


data.head(10)


# ## Data preprocessing
# playtime_forever: output\
# is_free: boolean => map 0/1\
# price: discrete number => scaling\
# genres: string => split and dummy (pivot)\
# categories: stirng => split and dummy (pivot)\
# tags: string => count its numbers of letter and catogorize\
# purchase_date, release_date: date => convert to stamptime and count the differece bewteen them as well as now\
# total_positive_reviews: float => scaling\
# total_negative_reviews: float => scaling

# In[201]:


# is_free: boolean => map 0/1
data['is_free'] = data['is_free'].map({False:0, True:1})


# In[202]:


# price: discrete number => scaling
data['price'].nlargest(5)


# In[203]:


# Top 2 is way larger and may have negative effect to the whole dataset. So will be removed.
data = data.drop(data['price'].nlargest(2).index)


# In[204]:


data['price'].describe()


# In[205]:


# It doesn't meet the std. So we scale it by MinMaxScaler
price_scaler = MinMaxScaler()
price_scaler.fit(data['price'].values.reshape(-1,1))
price_scaled = price_scaler.fit_transform(data['price'].values.reshape(-1,1))


# In[206]:


data['price'] = price_scaled


# In[207]:


# genres: string => split and dummy (pivot)
data['genres'].isna().sum()


# In[208]:


genres_split = data['genres'].apply(lambda x: x.strip().split(","))


# In[209]:


genres = pd.get_dummies(genres_split.apply(pd.Series).stack()).sum(level=0)


# In[210]:


genres.columns = pd.MultiIndex.from_product([['genres'], genres.columns])


# In[211]:


data = data.drop(columns=['genres'])


# In[212]:


data = pd.concat([data, genres], axis=1)


# In[213]:


# categories: stirng => split and dummy (pivot)
data['categories'].isna().sum()


# In[214]:


categories_split = data['categories'].apply(lambda x: x.strip().split(","))
categories = pd.get_dummies(categories_split.apply(pd.Series).stack()).sum(level=0)
categories.columns = pd.MultiIndex.from_product([['categories'], categories.columns])
data = data.drop(columns=['categories'])
data = pd.concat([data, categories], axis=1)


# In[215]:


# tags: string => count its numbers of letter and catogorize
data['tags'].isna().sum()


# In[216]:


data['tags'] = data['tags'].apply(len)


# In[217]:


data['tags'].describe()


# In[218]:


data.loc[data['tags'] <= 170, 'tags'] = 0
data.loc[(data['tags'] > 170) & (data['tags'] <= 197), 'tags'] = 0.25
data.loc[(data['tags'] > 197) & (data['tags'] <= 213), 'tags'] = 0.5
data.loc[data['tags'] > 213, 'tags'] = 1


# In[219]:


# purchase_date, release_date: date => convert to stamptime and count the differece bewteen them as well as now
data['purchase_date'].isna().sum()


# In[220]:


# Drop the examples with NaN value
data = data.drop(data.loc[data['purchase_date'].isna(), :].index)


# In[221]:


data['purchase_date'] = data['purchase_date'].apply(lambda x: datetime.strptime(x, '%b %d, %Y'))


# In[222]:


data['release_date'].isna().sum()


# In[223]:


data.loc[data['release_date'] == 'Nov 10, 2016', 'release_date'] = '10 Nov, 2016' # Exception 


# In[224]:


data['release_date'] = data['release_date'].apply(lambda x: datetime.strptime(x, '%d %b, %Y'))


# In[225]:


data['day_btn_purchase_release'] = (data['purchase_date'] - data['release_date']).dt.days


# In[226]:


data['day_btn_2020_purchase'] = (datetime(2020,1,1)-data['purchase_date']).dt.days


# In[227]:


# scaling day
data['day_btn_purchase_release'].max()


# In[228]:


data['day_btn_purchase_release'].min()


# In[229]:


# data['day_btn_purchase_release'] divided by 5000
data['day_btn_purchase_release'] = data['day_btn_purchase_release'] / 5000


# In[230]:


data['day_btn_2020_purchase'].max()


# In[231]:


data['day_btn_2020_purchase'].min()


# In[232]:


# data['day_btn_2020_purchase'] divided by 2000
data['day_btn_2020_purchase'] = data['day_btn_2020_purchase'] / 2000


# In[233]:


# Remove purchase_date and release_date
data = data.drop(columns=['purchase_date', 'release_date'])


# In[234]:


# Inspect the correlation bewteen the play time and positive comment
plt.scatter(x=data['total_positive_reviews'], y=data['playtime_forever'])


# In[235]:


plt.scatter(x=data['total_negative_reviews'], y=data['playtime_forever'])


# In[236]:


# So just scalling
data['total_positive_reviews'].max()


# In[237]:


# Divided by 100000
data['total_positive_reviews'] = data['total_positive_reviews'] / 100000


# In[238]:


data['total_negative_reviews'].max()


# In[239]:


# Divided by 100000
data['total_negative_reviews'] = data['total_negative_reviews'] / 100000


# In[240]:


# Save the prepocsssing data
data.to_csv('prepocessed_data.csv')


# ## Data split

# In[241]:


X = data.drop(columns=['playtime_forever'])
y = data['playtime_forever']


# In[242]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


# ## Train Model

# In[243]:


svr = svm.SVR()
svr.fit(X_train, y_train) 


# In[244]:


mlp = MLPRegressor(batch_size=2, max_iter=10000)
mlp.fit(X_train, y_train) 


# In[245]:


ada = AdaBoostRegressor(random_state=0, n_estimators=100)
ada.fit(X_train, y_train)  


# In[256]:


rdf = RandomForestRegressor()
rdf.fit(X_train, y_train)


# In[247]:


knn = KNeighborsRegressor()
knn.fit(X_train, y_train)


# In[248]:


xgb = XGBRegressor(n_estimators=1000, learning_rate=0.1)
xgb.fit(X_train, y_train)


# ## Evaluate the models

# In[249]:


np.sqrt(mean_squared_error(svr.predict(X_test), y_test))


# In[250]:


np.sqrt(mean_squared_error(mlp.predict(X_test), y_test))


# In[251]:


np.sqrt(mean_squared_error(ada.predict(X_test), y_test))


# In[257]:


np.sqrt(mean_squared_error(rdf.predict(X_test), y_test))


# In[253]:


np.sqrt(mean_squared_error(knn.predict(X_test), y_test))


# In[254]:


np.sqrt(mean_squared_error(xgb.predict(X_test), y_test))


# In[189]:


vonderland_xgb = XGBRegressor(n_estimators=100,
                    learning_rate = .01,
                    max_depth = 4,
                    random_state=42,
                    n_jobs = -1,
                    early_stopping_rounds=10)
vonderland_xgb.fit(X_train, y_train)
np.sqrt(mean_squared_error(vonderland_xgb.predict(X_test), y_test))


# In[190]:


vonderland_rdf = RandomForestRegressor(n_estimators=60, oob_score=True, random_state=1)
vonderland_rdf.fit(X_train, y_train)
np.sqrt(mean_squared_error(vonderland_rdf.predict(X_test), y_test))


# ## Cross Validation

# In[255]:


-1 * cross_val_score(svr, X, y,
                              cv=5,
                              scoring='neg_mean_squared_error').mean()


# In[191]:


-1 * cross_val_score(vonderland_xgb, X, y,
                              cv=5,
                              scoring='neg_mean_squared_error').mean()


# In[192]:


-1 * cross_val_score(vonderland_rdf, X, y,
                              cv=5,
                              scoring='neg_mean_squared_error').mean()


# In[193]:


-1 * cross_val_score(xgb, X, y,
                              cv=5,
                              scoring='neg_mean_squared_error').mean()


# In[272]:


rdf = RandomForestRegressor()
-1 * cross_val_score(rdf, X, y,
                              cv=10,
                              scoring='neg_mean_squared_error').mean()


# In[273]:


rdf = RandomForestRegressor(n_estimators=56)
-1 * cross_val_score(rdf, X, y,
                              cv=10,
                              scoring='neg_mean_squared_error').mean()


# In[276]:


rdf = RandomForestRegressor(n_estimators=56,)
-1 * cross_val_score(rdf, X, y,
                              cv=10,
                              scoring='neg_mean_squared_error').mean()


# ## Hyperparameter tuning

# In[324]:


# Random forest


# In[286]:


from sklearn.model_selection import RandomizedSearchCV# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 56, stop = 1120, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
rdf = RandomForestRegressor()
rdf_random = RandomizedSearchCV(estimator = rdf, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = -1)

# Fit the random search model
rdf_random.fit(X_train, y_train)


# In[287]:


rdf_random.best_params_


# In[288]:


rdf = RandomForestRegressor(n_estimators=174, 
                            min_samples_split=10,
                            min_samples_leaf=4,
                            max_features='sqrt', 
                            max_depth=90, 
                            bootstrap=True)
-1 * cross_val_score(rdf, X, y,
                              cv=10,
                              scoring='neg_mean_squared_error').mean()


# In[378]:


# XGBoost
from sklearn.model_selection import RandomizedSearchCV# Number of trees in random forest

random_grid = {'n_estimators': [int(x) for x in np.linspace(start = 100, stop = 1000, num = 100)],
               'learning_rate': [0.1, 0.01, 0.001],
               'max_depth': [3,4,5],
               'objective': ['reg:squarederror'],
               }
xgb = XGBRegressor()
xgb_random = RandomizedSearchCV(estimator = xgb, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = -1)

# Fit the random search model
xgb_random.fit(X, y)


# In[379]:


xgb_random.best_params_


# In[380]:


xgb = XGBRegressor(n_estimators=609, objective='reg:squarederror', max_depth=4, learning_rate=0.001)
-1 * cross_val_score(xgb, X, y,
                              cv=10,
                              scoring='neg_mean_squared_error').mean()


# In[76]:


data.shape


# In[77]:


test_data = pd.read_csv('test.csv', index_col='id')


# ## Data preprocessing for test data
# playtime_forever: output\
# is_free: boolean => map 0/1\
# price: discrete number => scaling\
# genres: string => split and dummy (pivot)\
# categories: stirng => split and dummy (pivot)\
# tags: string => count its numbers of letter and catogorize\
# purchase_date, release_date: date => convert to stamptime and count the differece bewteen them as well as now\
# total_positive_reviews: float => scaling\
# total_negative_reviews: float => scaling

# In[79]:


# is_free: boolean => map 0/1
test_data['is_free'] = test_data['is_free'].map({False:0, True:1})


# In[80]:


# price: discrete number => scaling
test_data['price'] = test_data['price'] / 30800


# In[81]:


# genres: string => split and dummy (pivot)
genres_split = test_data['genres'].apply(lambda x: x.strip().split(","))
genres = pd.get_dummies(genres_split.apply(pd.Series).stack()).sum(level=0)
genres.columns = pd.MultiIndex.from_product([['genres'], genres.columns])
test_data = test_data.drop(columns=['genres'])
test_data = pd.concat([test_data, genres], axis=1)


# In[82]:


# categories: stirng => split and dummy (pivot)
categories_split = test_data['categories'].apply(lambda x: x.strip().split(","))
categories = pd.get_dummies(categories_split.apply(pd.Series).stack()).sum(level=0)
categories.columns = pd.MultiIndex.from_product([['categories'], categories.columns])
test_data = test_data.drop(columns=['categories'])
test_data = pd.concat([test_data, categories], axis=1)


# In[83]:


# tags: string => count its numbers of letter and catogorize
test_data['tags'] = test_data['tags'].apply(len)
test_data.loc[test_data['tags'] <= 170, 'tags'] = 0
test_data.loc[(test_data['tags'] > 170) & (test_data['tags'] <= 197), 'tags'] = 1
test_data.loc[(test_data['tags'] > 197) & (test_data['tags'] <= 213), 'tags'] = 2
test_data.loc[test_data['tags'] > 213, 'tags'] = 3


# In[84]:


# purchase_date, release_date: date => convert to stamptime and count the differece bewteen them as well as now
test_data['purchase_date'] = test_data['purchase_date'].fillna("Sep 2, 2019")
test_data['purchase_date'] = test_data['purchase_date'].apply(lambda x: datetime.strptime(x, '%b %d, %Y'))
test_data['release_date'] = test_data['release_date'].apply(lambda x: datetime.strptime(x, '%d-%b-%y'))
test_data['day_btn_purchase_release'] = (test_data['purchase_date'] - test_data['release_date']).dt.days
test_data['day_btn_2020_purchase'] = (datetime(2020,1,1)-test_data['purchase_date']).dt.days
test_data['day_btn_purchase_release'] = test_data['day_btn_purchase_release'] / 5000
test_data['day_btn_2020_purchase'] = test_data['day_btn_2020_purchase'] / 2000
test_data = test_data.drop(columns=['purchase_date', 'release_date'])


# In[85]:


# total_positive_reviews: float => scaling
# total_negative_reviews: float => scaling
test_data['total_positive_reviews'] = test_data['total_positive_reviews'].fillna(0)
test_data['total_negative_reviews'] = test_data['total_negative_reviews'].fillna(0)
test_data['total_positive_reviews'] = test_data['total_positive_reviews'] / 100000
test_data['total_negative_reviews'] = test_data['total_negative_reviews'] / 100000


# In[86]:


test_data[('genres', 'Racing')] = 0
test_data[('genres', 'Design & Illustration')] = 0
test_data[('genres', 'Utilities')] = 0
test_data[('genres', 'Sexual Content')] = 0
test_data[('categories', 'Valve Anti-Cheat enabled')] = 0
test_data[('genres', 'Animation & Modeling')] = 0
test_data[('genres', 'Audio Production')] = 0
test_data['playtime_forever'] = 0


# In[87]:


test_data = test_data[data.columns]
test_data = test_data.drop(columns=['playtime_forever'])


# In[ ]:





# ## Random Forest

# In[347]:


rdf = RandomForestRegressor(n_estimators=174, 
                            min_samples_split=10,
                            min_samples_leaf=4,
                            max_features='sqrt', 
                            max_depth=90, 
                            bootstrap=True)
rdf.fit(X, y)


# In[290]:


# Use random forest
playtime_forever = rdf.predict(test_data)
submission = pd.DataFrame(data=playtime_forever, columns=['playtime_forever'])
submission.index.name = 'id'
submission.to_csv('rdf.csv')
#16
-1 * cross_val_score(rdf, X, y,
                              cv=10,
                              scoring='neg_mean_squared_error').mean()


# In[349]:





# ## Neural Network

# In[353]:


# Neural Network
random_grid = {'hidden_layer_sizes': [(50,), (100,), (200,)],
               'solver': ['lbfgs', 'adam'],
               'alpha': [0.001, 0.0001],
               'learning_rate_init': [0.01, 0.001, 0.0001]
               }
mlp = MLPRegressor()
mlp_random = RandomizedSearchCV(estimator = mlp, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = -1)

# Fit the random search model
mlp_random.fit(X, y)


# In[354]:


mlp_random.best_params_


# In[382]:


mlp = MLPRegressor(solver= 'adam', learning_rate_init= 0.0001, hidden_layer_sizes= (200,), alpha= 0.0001)
mlp.fit(X,y)

playtime_forever = mlp.predict(test_data)
submission = pd.DataFrame(data=playtime_forever, columns=['playtime_forever'])
submission.index.name = 'id'
submission.to_csv('mlp.csv')
#16
-1 * cross_val_score(mlp, X, y,
                              cv=10,
                              scoring='neg_mean_squared_error').mean()


# In[383]:


mlp = MLPRegressor(solver= 'adam', learning_rate_init= 0.0001, hidden_layer_sizes= (200,), alpha= 0.0001)
mlp.fit(X,y)
mlp.score(X,y)


# ## See the features importance of random forest

# In[335]:


feature_list = list(X.columns)
# Get numerical feature importances
importances = list(rdf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
for pair in feature_importances:
    print("Variable: " + str(pair[0]) + " Importance: " + str(pair[1]))


# In[336]:


plt.figure(figsize=(16,9))
# Reset style 
plt.style.use('fivethirtyeight')

# list of x locations for plotting
x_values = list(range(len(importances)))

# Make a bar chart
plt.bar(x_values, importances, orientation = 'vertical', color = 'r', edgecolor = 'k', linewidth = 1.2)

# Tick labels for x axis
plt.xticks(x_values, feature_list, rotation='vertical')

# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances of random forest');


# In[ ]:





# In[361]:


mlp.fit(X, y)


# In[258]:


# Use MLP
playtime_forever = rdf.predict(test_data)
submission = pd.DataFrame(data=playtime_forever, columns=['playtime_forever'])
submission.index.name = 'id'
submission.to_csv('mlp.csv')
#16


# In[259]:


# Use KNN
knn.fit(X, y)


# In[260]:


playtime_forever = knn.predict(test_data)
submission = pd.DataFrame(data=playtime_forever, columns=['playtime_forever'])
submission.index.name = 'id'
submission.to_csv('knn.csv')
#17


# In[262]:


# Use SVR
svr.fit(X, y)


# In[263]:


playtime_forever = svr.predict(test_data)
submission = pd.DataFrame(data=playtime_forever, columns=['playtime_forever'])
submission.index.name = 'id'
submission.to_csv('svr.csv')
#19


# In[340]:


# Use XGBoost
xgb.fit(X, y)


# In[341]:


playtime_forever = xgb.predict(test_data)
submission = pd.DataFrame(data=playtime_forever, columns=['playtime_forever'])
submission.index.name = 'id'
submission.to_csv('xgb.csv')


# In[342]:


xgb.score(X,y)


# ## See the features importance of random forest

# In[337]:


feature_list = list(X.columns)
# Get numerical feature importances
importances = list(xgb.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
for pair in feature_importances:
    print("Variable: " + str(pair[0]) + " Importance: " + str(pair[1]))


# In[338]:


plt.figure(figsize=(16,9))
# Reset style 
plt.style.use('fivethirtyeight')

# list of x locations for plotting
x_values = list(range(len(importances)))

# Make a bar chart
plt.bar(x_values, importances, orientation = 'vertical', color = 'r', edgecolor = 'k', linewidth = 1.2)

# Tick labels for x axis
plt.xticks(x_values, feature_list, rotation='vertical')

# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances for xgboost');


# In[351]:


data['playtime_forever'].mean()


# In[ ]:




