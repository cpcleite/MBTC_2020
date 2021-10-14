# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 13:14:22 2020

@author: cpcle
"""

# import required libraries
#import numpy                as np
import pandas               as pd
#import matplotlib.pyplot    as plt

from sklearn.model_selection  import train_test_split, GridSearchCV,\
                                     StratifiedKFold
from sklearn.preprocessing    import OneHotEncoder
from sklearn.metrics          import balanced_accuracy_score,\
                                     confusion_matrix, roc_auc_score
# Import Forest Tree Classifer
from sklearn.ensemble         import RandomForestClassifier
from sklearn.pipeline         import Pipeline

#from imblearn.over_sampling   import SMOTE
#from imblearn.under_sampling  import RandomUnderSampler
#from imblearn.pipeline        import Pipeline

# Load csv file
df = pd.read_csv('algar-dataset-treino.csv', sep=',')

#print(df.info())
#print(df['Pontuação teste'].describe())

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=555)

# Instantiate the classifier: ftc
ftc = RandomForestClassifier()

param_grid ={
              'n_estimators' : [80, 100, 120],
              'max_depth'    : [None, 1, 2],
              'class_weight' :[{'Sim': 2, 'Não':1},
                               {'Sim': 1, 'Não':2}] 
}

# Perform grid search: grid_mse
grid_acc = GridSearchCV(param_grid=param_grid, estimator=ftc,
                        scoring = "balanced_accuracy", cv=5, verbose=1,
                        n_jobs=2)

# Fit grid_mse to the data
grid_acc.fit(X_train, y_train)

# Print the best parameters and lowest RMSE
print("\nBest parameters found: ", grid_acc.best_params_)
print("\nHighest average balanced accuracy found: %.4f" % grid_acc.best_score_)

print('\nBest Estimator: \n', grid_acc.best_estimator_)

print(' \nBest test score: %f' % (grid_acc.score(X_test, y_test)))

#Print scores
print('\nTrain score: %f' % grid_acc.score(X_train, y_train))
print('Test  score: %f\n' % grid_acc.score(X_test, y_test))

y_pred = grid_acc.predict(X_test)
ba = balanced_accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred, labels=['Não', 'Sim'])
ac = cm[0,0]/(cm[0,0]+cm[0,1])
sp = cm[1,1]/(cm[1,1]+cm[1,0])
auc = roc_auc_score(y_test=='Não', y_pred=='Não')

print('\nBalanced Accuracy : %.4f' % ba)
print('AUC               : %.4f' % auc)
print('Accuracy          : %.4f' % ac)
print('Specificity       : %.4f\n' % sp)

#Best parameters found:  {'SMOTE__sampling_strategy': 0.3,
#                         'model__colsample_bytree': 0.055,
#                         'model__learning_rate': 0.5,
#                         'model__max_depth': 1,
#                         'model__n_estimators': 130,
#                         'model__reg_alpha': 0.25,
#                         'model__reg_gamma': 0.1,
#                         'under__sampling_strategy': 0.7}

#Highest average balanced accuracy found: 0.7688

#Best Estimator: 
# Pipeline(steps=[('SMOTE', SMOTE(sampling_strategy=0.3)),
#                ('under',
#                 RandomUnderSampler(random_state=555, sampling_strategy=0.7)),
#                ('model',
#                 XGBClassifier(colsample_bytree=0.055, learning_rate=0.5,
#                               max_depth=1, n_estimators=130, reg_alpha=0.25,
#                               reg_gamma=0.1))])
 
#Best test score: 0.768953

#Train score: 0.802927
#Test  score: 0.768953


#Balanced Accuracy : 0.7690
#AUC               : 0.7690
#Accuracy          : 0.9100
#Specificity       : 0.6279

#[Parallel(n_jobs=4)]: Done 25920 out of 25920 | elapsed: 15.4min finished