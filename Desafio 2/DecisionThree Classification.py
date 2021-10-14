# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 13:28:22 2020

@author: cpcle
"""

# Import numpy and pandas
import numpy      as np
import pandas     as pd

# Import Classifiers
import xgboost    as xgb
from sklearn.tree            import DecisionTreeClassifier

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
df = pd.DataFrame.from_records(d1.transform(X=df))

si = SimpleImputer(
    missing_values=np.nan,  # os valores faltantes são do tipo ``np.nan`` (padrão Pandas)
    strategy='constant',  # a estratégia escolhida é a alteração do valor faltante por uma constante
    fill_value=0,  # a constante que será usada para preenchimento dos valores faltantes é um int64=0.
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
dt_cl = DecisionTreeClassifier()

# Fit the classifier to the training set
dt_cl.fit(X_train, y_train)

# Predict the labels of the test set: preds
preds = dt_cl.predict(X_test)

# Compute the accuracy: accuracy
accuracy = float(np.sum(preds==y_test))/y_test.shape[0]
print("Non tunned test accuracy: %f\n" % (accuracy))

# tunning parameters
gbm_param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_features': [None, 'auto', 'sqrt', 'log2'],
    'max_depth': [None, 7, 5]
}

# Instantiate the regressor: gbm
dtc = DecisionTreeClassifier()

# Perform grid search: grid_mse
grid_acc = GridSearchCV(param_grid=gbm_param_grid, estimator=dtc,
                        scoring = "accuracy", cv=4, verbose=1)

# Fit grid_mse to the data
grid_acc.fit(X_train, y_train)

# Print the best parameters and lowest RMSE
print("Best parameters found: ", grid_acc.best_params_)
print("\nHighest Accuracy found: ", grid_acc.best_score_)

print('\nBest Estimator: \n', grid_acc.best_estimator_)

print(' \nBest test score:%f' % (grid_acc.score(X_test, y_test)))

dtc = DecisionTreeClassifier(criterion='gini',
                        max_depth=7,
                        max_features=None)

#Fit model
dtc.fit(X_train, y_train)

#Print scores
print('\nTrain score: %f:' % dtc.score(X_train, y_train))
print('\nTest score: %f' % dtc.score(X_test, y_test))

