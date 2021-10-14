# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 08:46:57 2020

@author: Celso Leite
"""

#%% Preparação

# import relevant libraries
import pandas                 as pd
import numpy                  as np
import matplotlib.pyplot      as plt

from sklearn.model_selection  import train_test_split, GridSearchCV,\
                                     StratifiedKFold
                                     
from sklearn.preprocessing    import OneHotEncoder, StandardScaler
from sklearn.metrics          import f1_score, accuracy_score,\
                                     confusion_matrix, roc_auc_score,\
                                     plot_confusion_matrix
# Balanceamento das categorias
from imblearn.over_sampling   import SMOTE
from imblearn.under_sampling  import RandomUnderSampler

# Algoritmo utilizado
from keras.models             import Sequential
from keras.layers             import Dense

# Lê arquivo de dados
df = pd.read_csv('leitura_iot.csv')
df.drop_duplicates(inplace=True)
    
# Lista final do datatset
#print(df.tail())

# Informações do DataSet
#print(df.info())

# Quantidade de NaN's por coluna
#print(df.isna().sum())

# Distribuição dos categóricos
#print(df.nunique())

# Datas com problemas
#print(df[pd.to_datetime(df['Tempo'], format='%Y-%m-%d',
#                        errors='coerce').isna()]['Tempo'].value_counts())

#prods = ['Original_473', 'Original_269', 'Zero', 'Maçã-Verde',
#            'Tangerina', 'Citrus', 'Açaí-Guaraná', 'Pêssego']

#a = df.melt(id_vars = ['Tempo'],
#            value_vars=prods,
#            value_name='Venda',
#            var_name='Produto')

#a['Data'] = pd.to_datetime(a['Tempo'], format='%Y-%m-%d', errors='coerce')

#df['Data'] = pd.to_datetime(df['Tempo'], format='%Y-%m-%d',
#                            errors='coerce')

#df.set_index('Data', inplace=True)
#a = df[prods].groupby(pd.Grouper(freq="M")).sum()

#df['Original_473'].plot.hist(bins=8)
#plt.show()

#df.groupby('Estação').sum()[prods].sum(axis=1).plot.bar()
#plt.show()

#del [prods, a]

# Lista de colunas não utilizadas
drop_cols = ['LAT', 'LONG', 'Movimentação', 'Tempo', 'Data']

# Cria lista das colunas categóricas
col_cat = [col for col in df.nunique().index if  (df[col].dtype == 'O')
                                               & (col not in
                                                    ['TARGET', 'Tempo'])
           ]

# Busca categorias das colunas categóricas
categorias = {}
for col in col_cat:    
    categorias[col]= list(df[col].value_counts().index)
    
# Lista de colunas categóricas a codificar
col_cat = [col for col in col_cat if col not in drop_cols]

# Lista colunas numéricas
col_num = [col for col in df.columns if col not in \
                                 (col_cat + drop_cols + ['TARGET', 'Tempo'])]
    
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state = 123)

# Apaga variáveis já utilizadas
del [df, X, y, col]

#%% Prepara features

# Gera lista de categorias a codificar
lista_cat = [categorias[col] for col in col_cat]

#%% Transformação das features

features = X_train.copy()

# Coloca colunas das distâncias
#features = features.merge(Distancias, how='inner', left_on=['Estação'],
#                          right_index=True)

#col_num.extend(Distancias.columns.tolist())

# Cria colunas de tempo

# Substitui 2019-2-29
features['Tempo'].replace({'2018-2-29' : '2018-2-28',
                           '2018-2-30' : '2018-2-18',
                           '2019-2-29' : '2019-2-28',
                           '2019-2-30' : '2019-2-28',
                           },
                          inplace=True)

features['Data'] = pd.to_datetime(features['Tempo'], format='%Y-%m-%d')

# Cria colunas de Datas
features['Dia'] = features['Data'].dt.day
features['Semana'] = features['Data'].dt.isocalendar().week.astype('int64')
features['DiaSemana'] = features['Data'].dt.weekday
features['Mes'] = features['Data'].dt.month
features['AnoMes'] = features['Data'].dt.strftime('%Y%m').astype('int64')

col_num.extend(['Dia', 'Semana', 'DiaSemana', 'Mes', 'AnoMes'])
                
# Cria coluna com estoque mínimo
prods = ['Original_473', 'Original_269', 'Zero', 'Maçã-Verde',
         'Tangerina', 'Citrus', 'Açaí-Guaraná', 'Pêssego']
        
scaler = StandardScaler()
features[prods] = scaler.fit_transform(features[prods])
features['Mínimo'] = features[prods].apply(np.min, axis='columns')
col_num.extend(['Mínimo'])

# Elimina colunas não utilizadas
features.drop(columns = drop_cols, inplace= True)

# Separação em colunas categóricas e numéricas                                   
categorical_columns = features.loc[:, col_cat]
numeric_columns     = features.loc[:, col_num]

# Codifica Colunas Categóricas
ohe = OneHotEncoder(handle_unknown='ignore', categories=lista_cat)

a = ohe.fit_transform(categorical_columns)
b = pd.DataFrame(a.toarray())
b.columns = ohe.get_feature_names().tolist()
b.index = categorical_columns.index

features = pd.concat([b, 
                    numeric_columns], ignore_index=False,
                    axis='columns')
del [a, b]
#%% Parametriza o Modelo

# Balanceamento
features, y_train = SMOTE(sampling_strategy='auto').fit_resample(features, y_train)
features, y_train = RandomUnderSampler(sampling_strategy='auto', random_state=555).fit_resample(features, y_train)
kfold = StratifiedKFold(n_splits=5, shuffle=False)

model = xgb.XGBClassifier(booster='gbtree', objective='binary:logistic',
                          eval_metric='merror', n_estimators= 100,
                          max_depth=6, min_child_weight=0.25, gamma=0,
                          subsample=1, colsample_bytree=1, eta=0.3)

parametros = { 'n_estimators'     : [20, 50, 100, 150],
               
    }

# Perform grid search: grid_mse
grdcv = GridSearchCV(param_grid=parametros, estimator=model,
                        scoring = 'f1_weighted', cv=kfold, verbose=1,
                        n_jobs=4)

#grdcv = xgb.XGBClassifier(booster='gbtree',
#                          n_estimators=200, eta=0.05, max_depth=6,
#                          min_child_weight=2, gamma=0.5, subsample=1,
#                          colsample_bytree=1, objective='multi:softmax',
#                          eval_metric='merror', num_class=6, n_jobs=4)

#%% Treina Modelo

# Fit the classifier to the training set
grdcv.fit(features, y_train)

print('\nBest parameters :', grdcv.best_params_)
print('\nBest score : %.4f\n' % grdcv.best_score_)

#%% Training Prediction

# Predict the labels of the test set: preds
y_pred = grdcv.predict(features)

# Compute the accuracy:
accuracy = accuracy_score(y_train, y_pred)
f1 = f1_score(y_train, y_pred, average='weighted')
cm = confusion_matrix(y_train, y_pred, labels=['NORMAL', 'REABASTECER'])
ac = cm[0,0]/(cm[0,0]+cm[0,1])
sp = cm[1,1]/(cm[1,1]+cm[1,0])
se = cm[0,0]/(cm[0,0]+cm[1,0])
auc = roc_auc_score(y_train=='NORMAL', y_pred=='NORMAL')

print("Train accuracy    : %.4f" % (accuracy))
print("Train F1_score    : %.4f" % (f1))
print('Sensitivity       : %.4f' % se)
print('AUC               : %.4f' % auc)
print('Specificity       : %.4f' % sp)
print('F1 weighted       : %.4f\n' % f1)


#%% Prepara features de teste

features = X_test.copy()

# Coloca colunas das distâncias
#features = features.merge(Distancias, how='inner', left_on=['Estação'],
#                          right_index=True)

# Cria colunas de tempo

# Substitui 2019-2-29
features['Tempo'].replace({'2018-2-29' : '2018-2-28',
                           '2018-2-30' : '2018-2-18',
                           '2019-2-29' : '2019-2-28',
                           '2019-2-30' : '2019-2-28',
                           },
                          inplace=True)

# Cria colunas de datas
features['Data'] = pd.to_datetime(features['Tempo'], format='%Y-%m-%d')
features['Dia'] = features['Data'].dt.day
features['Semana'] = features['Data'].dt.isocalendar().week.astype('int64')
features['DiaSemana'] = features['Data'].dt.weekday
features['Mes'] = features['Data'].dt.month
features['AnoMes'] = features['Data'].dt.strftime('%Y%m').astype('int64')

# Cria coluna com estoque mínimo
features[prods] = scaler.transform(features[prods])
features['Mínimo'] = features[prods].apply(np.min, axis='columns')

# Elimina colunas não utilizadas
features.drop(columns = drop_cols, inplace= True)
                                   
# Separação em colunas categóricas e numéricas                
categorical_columns = features.loc[:, col_cat]
numeric_columns     = features.loc[:, col_num]

# Codifica Colunas Categóricas
a = ohe.transform(categorical_columns)
b = pd.DataFrame(a.toarray())
b.columns = ohe.get_feature_names().tolist()
b.index = categorical_columns.index

features = pd.concat([b, 
                    numeric_columns], ignore_index=False,
                    axis='columns')

del [a, b]

#%% Avalia Modelo

# Predict the labels of the test set: preds
y_pred = grdcv.predict(features)

# Compute the accuracy:
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
cm = confusion_matrix(y_test, y_pred, labels=['NORMAL', 'REABASTECER'])
ac = cm[0,0]/(cm[0,0]+cm[0,1])
sp = cm[1,1]/(cm[1,1]+cm[1,0])
se = cm[0,0]/(cm[0,0]+cm[1,0])
auc = roc_auc_score(y_test=='NORMAL', y_pred=='NORMAL')

print("Test accuracy     : %.4f" % (accuracy))
print("Test F1_score     : %.4f" % (f1))
print('Test Sensitivity  : %.4f' % se)
print('Test AUC          : %.4f' % auc)
print('Test Specificity  : %.4f' % sp)
print('Test F1 weighted  : %.4f\n' % f1)

plot_confusion_matrix(grdcv, features, y_test)
plt.savefig('CV.png')
plt.show()

pd.DataFrame(grdcv.cv_results_).to_excel('results.xlsx')

c = pd.DataFrame(grdcv.best_estimator_.feature_importances_)
     
cols = features.columns.tolist()
                                 
cols.extend(col_num)

c = pd.concat([pd.DataFrame(cols), c], axis=1, ignore_index=True)
    
c.columns = ['coluna', 'importancia']
c.sort_values(['importancia'], ascending=False, inplace=True)