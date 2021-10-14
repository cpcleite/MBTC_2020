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

from sklearn.preprocessing    import OneHotEncoder, StandardScaler
from sklearn.metrics          import f1_score, accuracy_score,\
                                     confusion_matrix, roc_auc_score,\
                                     plot_confusion_matrix
# Algoritmo utilizado
from keras.models import Sequential
from keras.layers import Dense

# Lê arquivo de dados
df = pd.read_csv('leitura_iot.csv')
df.drop_duplicates(inplace=True)

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

# Apaga variáveis já utilizadas
del [df, col]

#%% Prepara features

# Gera lista de categorias a codificar
lista_cat = [categorias[col] for col in col_cat]

#%% Transformação das features

features = X.copy()
y_train = y.apply(lambda x: 1 if x=='NORMAL' else 0)
y_train = pd.concat([y_train,
                     1- y_train],
                    axis = 1)

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

# model

model = Sequential()

# Input and 3 hidden layers
model.add(Dense(2000, input_shape=(features.shape[1],),
                activation='relu'))
model.add(Dense(2000, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(8,    activation='relu'))

# Output Layer
model.add(Dense(2, activation='softmax'))

# Compile model
model.compile(optimizer = 'adam', metrics=['accuracy'],
              loss='binary_crossentropy')


#%% Treina Modelo

# Train model
history = model.fit(features, y_train, epochs=10,
                    validation_split=0.2)

# Fit the classifier to the training set
model.fit(features, y_train)

#print('\nBest parameters :', grdcv.best_params_)
#print('\nBest score : %.4f\n' % grdcv.best_score_)

#%% Training Prediction

#plot_confusion_matrix(model, features, y_train)
#plt.savefig('CV.png')
#plt.show()

# Predict the labels of the test set: preds
y_train = y_train.iloc[:,0].values
y_pred = model.predict(features)[:,0].astype('int')

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
