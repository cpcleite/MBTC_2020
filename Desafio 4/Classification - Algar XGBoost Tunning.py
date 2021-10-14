# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 13:14:22 2020

@author: cpcle
"""

# import required libraries
#import numpy                as np
import pandas               as pd
import matplotlib.pyplot    as plt

from sklearn.model_selection  import train_test_split, GridSearchCV,\
                                     StratifiedKFold
from sklearn.preprocessing    import OneHotEncoder
from sklearn.metrics          import balanced_accuracy_score,\
                                     confusion_matrix, roc_auc_score,\
                                     plot_confusion_matrix,\
                                     f1_score
# Import xgboost
import xgboost              as xgb

from imblearn.over_sampling   import SMOTE
#from imblearn.under_sampling  import RandomUnderSampler
from imblearn.pipeline        import Pipeline

# Load csv file
df = pd.read_csv('train_dataset_algartech.csv', sep=',')

#print(df.info())
#rint(df['Pontuação teste'].describe())

#df['Pontuação teste'].hist(bins=15)
#plt.show()

# Get Features and target
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

# Columns with one value
a= [col for col in df.columns if df[col].value_counts().count()<2]
print('Columns with unique value:', a)

# Eliminate columns with unique feature values
X = X.drop(a + ['Subordinado'], axis=1)

# Categorical Columns
c_columns = X.select_dtypes(include='object')
nc_columns = [col for col in X.columns if col not in c_columns.columns]

ohe = OneHotEncoder(handle_unknown='ignore')

c_columns = pd.DataFrame(ohe.fit_transform(c_columns).toarray())
c_columns.columns = ohe.get_feature_names()

X = pd.concat([X[nc_columns], c_columns], axis=1)

# Train test split  75% / 25%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=555)

# Instantiate the XGBClassifier: xg_cl
xg_cl = xgb.XGBClassifier(objective='binary:logistic', booster='gbtree')

# tunning parameters
gbm_param_grid = [
    {
     'model__max_depth': [2, 5, 6],
     'model__min_child_weight': [0.3, 0.5, 1],
     'model__gamma': [0, 1, 3]
#     'model__colsample_bytree': [0.055],
#     'model__n_estimators': [150, 200, 250],
#     'model__eta': [0,25, 0.3, 0.35]
    }
]

# Instantiate the classifier: gbm
gbm = xgb.XGBClassifier()
over = SMOTE(sampling_strategy='auto')
#under = RandomUnderSampler(sampling_strategy='auto')
kfold = StratifiedKFold(n_splits=5, shuffle=False)

# Assamble Pipeline
steps = [('SMOTE', over),
#         ('under', under), 
         ('model', gbm)]

pipeline = Pipeline(steps=steps)

# Perform grid search: grid_mse
grdcv = GridSearchCV(param_grid=gbm_param_grid, estimator=pipeline,
                        scoring = "f1_weighted", cv=kfold, verbose=1,
                        n_jobs=4)

# Fit grid_mse to the data
grdcv.fit(X_train, y_train)

# Print the best parameters and lowest RMSE
print("\nBest parameters found: ", grdcv.best_params_)
print("\nHighest average balanced accuracy found: %.4f" % grdcv.best_score_)

print('\nBest Estimator: \n', grdcv.best_estimator_)

print(' \nBest test score: %f' % (grdcv.score(X_test, y_test)))

#Print scores
print('\nTrain score: %f' % grdcv.score(X_train, y_train))
print('Test  score: %f\n' % grdcv.score(X_test, y_test))

y_pred = grdcv.predict(X_test)
ba = balanced_accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred, labels=['Não', 'Sim'])
ac = cm[0,0]/(cm[0,0]+cm[0,1])
sp = cm[1,1]/(cm[1,1]+cm[1,0])
auc = roc_auc_score(y_test=='Não', y_pred=='Não')
f1 = f1_score(y_test, y_pred, average='weighted')

print('\nBalanced Accuracy : %.4f' % ba)
print('AUC               : %.4f' % auc)
print('Accuracy          : %.4f' % ac)
print('Specificity       : %.4f\n' % sp)
print('F1 weighted       : %.4f\n' % f1)

#%% Resultados

plot_confusion_matrix(grdcv, X_test, y_test)  # doctest: +SKIP
plt.show()

pd.DataFrame(grdcv.cv_results_).to_excel('results.xlsx')

c = pd.DataFrame(grdcv.best_estimator_._final_estimator.feature_importances_)
cols = X_test.columns  
                                 
c = pd.concat([pd.DataFrame(cols), c], axis=1, ignore_index=True)
    
c.columns = ['coluna', 'importancia']
c.sort_values(['importancia'], ascending=False, inplace=True)

