# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 08:46:57 2020

@author: Celso Leite
"""
#%% Preparação

# import relevant libraries
import pandas                 as pd
import matplotlib.pyplot      as plt

from sklearn.model_selection  import train_test_split, GridSearchCV,\
                                     StratifiedKFold
                                     
from sklearn.pipeline         import FeatureUnion
                                     
from sklearn.preprocessing    import OneHotEncoder, FunctionTransformer
from sklearn.metrics          import f1_score, accuracy_score,\
                                     plot_confusion_matrix
# Balanceamento das categorias
from imblearn.over_sampling   import SMOTE
from imblearn.under_sampling  import RandomUnderSampler
from imblearn.pipeline        import Pipeline

# Algoritmo utilizado
#import xgboost                as xgb
from sklearn.ensemble         import RandomForestClassifier

# Lë arquivo de dados
if 'df_training_dataset' not in globals():
    df_training_dataset = pd.read_csv(r'training_dataset.csv')
    
# Lista final do datatset
#print(df_training_dataset.tail())

# Informações do DataSet
#print(df_training_dataset.info())

# Quantidade de NaN's por coluna
#print(df_training_dataset.isna().sum())

# Distribuição dos categóricos
#print(df_training_dataset.nunique())

# Print columns with less than 30 categories
#a = df_training_dataset.nunique()
#a = [col for col in a.index if a.loc[col] < 30]
#for col in a:
#    print('\n' + col)
#    print(df_training_dataset[col].value_counts())
    
# How many grads without university information    
#print(df_training_dataset[df_training_dataset['graduacao'] == 'SEM FORMAÇÃO']\
#         ['universidade'].value_counts(dropna=False)
#     )

# How many nas in university
#print(df_training_dataset[df_training_dataset['graduacao'].isna()]\
#         ['universidade'].value_counts(dropna=False)
#     )
    
#print(df_training_dataset[ df_training_dataset['universidade'].isna() & 
#                          (df_training_dataset['graduacao'] == 'SEM FORMAÇÃO')]\
#         ['graduacao'].value_counts()
#     )
    
#print((~df_training_dataset.isna()).any(axis=1).sum())

#print((~df_training_dataset[['total_modulos',
#                             'modulos_iniciados',
#                             'modulos_finalizados']]
#                         .isna()).all(axis=1).count())

#print((~df_training_dataset[['total_modulos',
#                             'modulos_iniciados',
#                             'modulos_finalizados']]\
#                         .isna()).all(axis=1).sum())

# Lista de colunas não utilizadas
drop_cols = ['id', 'importante_ter_certificado']#,
#             'como_conheceu_lit', 'interesse_mba_lit',
#             'pretende_fazer_cursos_lit', 'horas_semanais_estudo']

# Cria lista das colunas categóricas
a = df_training_dataset.nunique()
col_cat = [col for col in a.index if  (df_training_dataset[col].dtype == 'O')
                                & (col != 'categoria')]

# Busca categorias das colunas categóricas
categorias = {}
for col in col_cat:    
    categorias[col]= list(df_training_dataset[col].value_counts().index)
    
# Lista de colunas categóricas a codificar
col_cat = [col for col in col_cat if col not in drop_cols]
# Lista colunas numéricas
col_num = [col for col in df_training_dataset.columns if col not in \
                                         (col_cat + drop_cols + ['categoria'])]

X = df_training_dataset.iloc[:,:-1]
y = df_training_dataset.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state = 123)

# Apaga variáveis já utilizadas
del [df_training_dataset, X, y, a]

#%% Prepara features    

# Preenche NaNs
valores = {'universidade':'Não Informada', 'graduacao':'Não Informada',
           'profissao':'Não Informada', 'organizacao':'Não Informada',
           'pretende_fazer_cursos_lit':0.0,
           'interesse_mba_lit':0.0,
           'horas_semanais_estudo': X_train['horas_semanais_estudo'].mean(),
           'como_conheceu_lit':'Outros',
           'total_modulos':X_train['total_modulos'].mean(),
           'modulos_iniciados':X_train['modulos_iniciados'].mean(),
           'modulos_finalizados':X_train['modulos_finalizados'].mean(),
           'certificados':X_train['certificados'].mean()
           }

# Adiciona categorias preenchidos
for col in categorias:
    if valores[col] not in categorias[col]:
        categorias[col].append(valores[col])       
        
# Gera lista de categorias a codificar
lista_cat = [categorias[col] for col in col_cat]

#%% Treinamento do modelo

drop_columns = FunctionTransformer(lambda x: x.drop(columns = drop_cols,
                                                    inplace= False),
                                   validate=False)
fill_columns = FunctionTransformer(lambda x: x.fillna(value = valores,
                                                      inplace= False),
                                   validate=False)

categorical_columns = FunctionTransformer(lambda x: x[col_cat],
                                          validate=False)

numeric_columns = FunctionTransformer(lambda x: x[col_num],
                                      validate=False)

over = SMOTE(sampling_strategy='auto')
under = RandomUnderSampler(sampling_strategy='auto', random_state=555)
kfold = StratifiedKFold(n_splits=4, shuffle=False)
ohe = OneHotEncoder(handle_unknown='ignore', categories=lista_cat)
model = RandomForestClassifier()

cat_features_steps = [('cc', categorical_columns),
                      ('ohe', ohe)]

union = FeatureUnion([('cf', Pipeline(cat_features_steps)),
                      ('nc', numeric_columns)
                      ])

# Assamble Pipeline
steps = [('dc'   , drop_columns),
         ('fc'   , fill_columns),
         ('un'   , union),
         ('SMT', over),
         ('under', under), 
         ('model', model)]

pipeline = Pipeline(steps=steps)

parametros = { 'model__n_estimators'     : [50, 100, 200],
               'model__max_depth'        : [None, 5,6,7],
               'model__class_weight'     : ['balanced']
    }

# Perform grid search: grid_mse
grdcv = GridSearchCV(param_grid=parametros, estimator=pipeline,
                        scoring = 'f1_weighted', cv=kfold, verbose=1,
                        n_jobs=4)

#grdcv = xgb.XGBClassifier(booster='gbtree',
#                          n_estimators=200, eta=0.05, max_depth=6,
#                          min_child_weight=2, gamma=0.5, subsample=1,
#                          colsample_bytree=1, objective='multi:softmax',
#                          eval_metric='merror', num_class=6, n_jobs=4)

# Fit the classifier to the training set
grdcv.fit(X_train, y_train)

print('\nBest parameters :', grdcv.best_params_)
print('\nBest score : %.4f\n' % grdcv.best_score_)

#%% Training Prediction
# Predict the labels of the test set: preds
preds = grdcv.predict(X_train)

# Compute the accuracy: accuracy
accuracy = accuracy_score(y_train, preds)
f1 = f1_score(y_train, preds, average='weighted')
print("Train accuracy: %f" % (accuracy))
print("Train F1_score: %f\n" % (f1))


#%% Prepara features de teste

# Elimina colunas
#X_test = X_test.copy().reset_index(drop=True)
#X_test.drop(columns=drop_cols, inplace=True)

# Substitui valores nulos
#X_test.fillna(value=valores, inplace=True)

# Separa colunas categóricas
#c_columns = X_test.loc[:, col_cat]
#nc_columns = [col for col in X_test.columns if col not in c_columns.columns]

#c_columns = pd.DataFrame(ohe.transform(c_columns).toarray())
#c_columns.columns = ohe.get_feature_names()

#X_test = pd.concat([X_test.loc[:, nc_columns], c_columns], axis=1)
#print(X_test.info())
#print(X_test.shape)

#del [c_columns, nc_columns]

#%% Avalia Modelo

# Predict the labels of the test set: preds
preds = grdcv.predict(X_test)

# Compute the accuracy: accuracy
accuracy = accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds, average='weighted')
print("Test accuracy : %f" % (accuracy))
print("Test F1_score : %f\n" % (f1))

plot_confusion_matrix(grdcv, X_test, y_test)  # doctest: +SKIP
plt.show()

pd.DataFrame(grdcv.cv_results_).to_excel('results.xlsx')

c = pd.DataFrame(grdcv.best_estimator_._final_estimator.feature_importances_)

grdcv.best_estimator_.named_steps['un'].transformer_list[0][1]\
     .named_steps['ohe'].get_feature_names()
     
cols = grdcv.best_estimator_.named_steps['un']\
                                 .transformer_list[0][1]\
                                 .named_steps['ohe']\
                                 .get_feature_names().tolist()
                                 
cols.extend(col_num)

c = pd.concat([pd.DataFrame(cols), c], axis=1, ignore_index=True)
    
c.columns = ['coluna', 'importancia']
c.sort_values(['importancia'], ascending=False, inplace=True)