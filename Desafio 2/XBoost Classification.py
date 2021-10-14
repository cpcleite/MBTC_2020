# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 14:07:56 2020

@author: cpcle
"""
# %% Imports
# Import numpy and pandas
import numpy      as np
import pandas     as pd

# Import xgboost
import xgboost    as xgb

# Import train_test_split and GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# SimpleImputer
from sklearn.impute import SimpleImputer

# DropColumns class for Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')
    
# Read dataset
df = pd.read_csv(r'C:\Users\cpcle\OneDrive\Documentos\Celso\Maratona Behind the Code 2020\Desafio 2\assets\data_asset\dataset_desafio_2.csv')

d1 = DropColumns(columns=['NOME'])
d1.fit(X=df)
df = d1.transform(X=df)

si = SimpleImputer(
    missing_values=np.nan,  # os valores faltantes são do tipo ``np.nan`` (padrão Pandas)
    strategy='most_frequent',  # a estratégia escolhida é a alteração do valor faltante por uma constante
    verbose=0,
    copy=True
)

si.fit(X=df)
cols = df.columns
df = pd.DataFrame.from_records(si.transform(X=df), columns=cols)


# Create arrays for the features and the target: X, y
X, y = df.iloc[:,:-1], df.iloc[:,-1]

# Create the training and test sets
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=123)

# Instantiate the XGBClassifier: xg_cl
xg_cl = xgb.XGBClassifier(objective='binary:logistic', n_estimators=10, seed=123)

# Fit the classifier to the training set
xg_cl.fit(X_train, y_train)

# Predict the labels of the test set: preds
preds = xg_cl.predict(X_test)

# Compute the accuracy: accuracy
accuracy = float(np.sum(preds==y_test))/y_test.shape[0]
print("Non tunned accuracy: %f\n" % (accuracy))

# tunning parameters
gbm_param_grid = {
    'colsample_bytree': [0.1, 0.5, 0.7],
    'n_estimators': [45, 75, 105],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.05, 0.1, 0.3 ]
}

# Instantiate the regressor: gbm
gbm = xgb.XGBClassifier()#objective='binary:logistic')

# Perform grid search: grid_mse
grid_acc = GridSearchCV(param_grid=gbm_param_grid, estimator=gbm,
                        scoring = "accuracy", cv=4, verbose=1)

# Fit grid_mse to the data
grid_acc.fit(X_train, y_train)

# Print the best parameters and lowest RMSE
print("Best parameters found: ", grid_acc.best_params_)
print("\nHighest Accuracy found: ", grid_acc.best_score_)

print('\nBest Estimator: \n', grid_acc.best_estimator_)

print(' \nBest test score:%f' % (grid_acc.score(X_test, y_test)))

gbm = xgb.XGBClassifier(objective='binary:logistic',
                        colsample_bytree=0.5,
                        learning_rate=0.1,
                        max_depth=4,
                        n_estimators=75)

#Fit model
gbm.fit(X_train, y_train)

#Print scores
print('\nTrain score: %f' % gbm.score(X_train, y_train))
print('\nTest score: %f' % gbm.score(X_test, y_test))


#Best test score:0.809500
#Train score: 0.826500
#Test score: 0.809500