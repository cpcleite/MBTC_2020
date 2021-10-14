# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 08:46:57 2020

@author: Celso Leite
"""
#%% Preparação

# import relevant libraries
import pandas                 as pd
import matplotlib.pyplot      as plt

from sklearn.model_selection  import train_test_split
                                     
from sklearn.preprocessing    import OneHotEncoder
from sklearn.metrics          import f1_score, accuracy_score,\
                                     plot_confusion_matrix
# Balanceamento das categorias
from imblearn.over_sampling   import SMOTE
from imblearn.under_sampling  import RandomUnderSampler

# Algoritmo utilizado
import xgboost                as xgb
#from sklearn.ensemble         import RandomForestClassifier
#from sklearn.neighbors         import KNeighborsClassifier

# Lë arquivo de dados
if 'df_training_dataset' not in globals():
    df_training_dataset = pd.read_csv(r'training_dataset.csv')
    
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

#%% Transformação das características

X_train.drop(columns = drop_cols, inplace= True)
X_train.fillna(value = valores, inplace= True)

categorical_columns = X_train.loc[:, col_cat]
numeric_columns = X_train.loc[:, col_num]

X_train=None

ohe = OneHotEncoder(handle_unknown='ignore', categories=lista_cat)

a = ohe.fit_transform(categorical_columns)
b = pd.DataFrame(a.toarray())
b.columns = ohe.get_feature_names().tolist()
b.index = categorical_columns.index

X_train = pd.concat([b, 
                    numeric_columns], ignore_index=False,
                    axis='columns')

#%% Train Model
X_train, y_train = SMOTE(sampling_strategy='auto').fit_resample(X_train, y_train)
X_train, y_train = RandomUnderSampler(sampling_strategy='auto', random_state=555).fit_resample(X_train, y_train)
model = xgb.XGBClassifier(booster='gbtree',
                          n_estimators=200, eta=0.05, max_depth=6,
                          min_child_weight=2, gamma=0.5, subsample=1,
                          colsample_bytree=1, objective='multi:softmax',
                          eval_metric='merror', num_class=6, n_jobs=4)

# Fit the classifier to the training set
model.fit(X_train, y_train)

#%% Training Prediction
# Predict the labels of the test set: preds
y_pred = model.predict(X_train)

# Compute the accuracy: accuracy
accuracy = accuracy_score(y_train, y_pred)
f1 = f1_score(y_train, y_pred, average='weighted')
print("Train accuracy: %f" % (accuracy))
print("Train F1_score: %f\n" % (f1))


#%% Transformação dos testes
X_test.drop(columns = drop_cols, inplace= True)
X_test.fillna(value = valores, inplace= True)

categorical_columns = X_test.loc[:, col_cat]
numeric_columns = X_test.loc[:, col_num]

ohe = OneHotEncoder(handle_unknown='ignore', categories=lista_cat)

a = ohe.fit_transform(categorical_columns)
b = pd.DataFrame(a.toarray())
b.columns = ohe.get_feature_names().tolist()
b.index = categorical_columns.index

X_test = pd.concat([b, 
                    numeric_columns], ignore_index=False,
                    axis='columns')

#%% Avalia Modelo

# Predict the labels of the test set: preds
y_pred = model.predict(X_test)

# Compute the accuracy: accuracy
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
print("Test accuracy : %f" % (accuracy))
print("Test F1_score : %f\n" % (f1))

plot_confusion_matrix(model, X_test, y_test)  # doctest: +SKIP
plt.show()
