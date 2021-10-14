# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 13:14:22 2020

@author: cpcle
"""

# import required libraries
import numpy                as np
import pandas               as pd
import matplotlib.pyplot    as plt


from sklearn.model_selection  import train_test_split
from sklearn.preprocessing    import OneHotEncoder

#from sklearn.pipeline         import FeatureUnion, Pipeline

# Import xgboost
import xgboost              as xgb


# Load csv file
df = pd.read_csv('algar-dataset-treino.csv', sep=',')

print(df.info())

print(df['Pontuação teste'].describe())

df['Pontuação teste'].hist(bins=15)
plt.show()

# Get Features and target
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

# Columns with one value
a= [col for col in df.columns if df[col].value_counts().count()<2]
print('Columns with unique value:', a)

# Eliminate columns with unique feature values
X = X[list([col for col in X.columns if col not in a])]

# Categorical Columns
c_columns = X.select_dtypes(include='object')
nc_columns = [col for col in X.columns if col not in c_columns.columns]

ohe = OneHotEncoder(handle_unknown='ignore')

c_columns = pd.DataFrame(ohe.fit_transform(c_columns).toarray())
c_columns.columns = ohe.get_feature_names()

X = pd.concat([X[nc_columns], c_columns], axis=1)

# Train test split  75% / 25%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=555)

# Instantiate the XGBClassifier: xg_cl
xg_cl = xgb.XGBClassifier(objective='binary:logistic', n_estimators=10, seed=123)

# Fit the classifier to the training set
xg_cl.fit(X_train, y_train)

# Predict the labels of the test set: preds
preds = xg_cl.predict(X_test)

# Compute the accuracy: accuracy
accuracy = float(np.sum(preds==y_test))/y_test.shape[0]
print("Non tunned accuracy: %f\n" % (accuracy))


