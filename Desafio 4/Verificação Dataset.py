# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 23:53:45 2020

@author: cpcle
"""

import pandas               as pd

# Load csv file
#df = pd.read_csv('algar-dataset-treino.csv', sep=',')
df = pd.read_csv('XGBoost 05 100.csv', sep=',')

a = [(x, max(df[x].apply(len))) for x in df.columns if df[x].dtype =='object']
print(a)

print(df.dtypes.value_counts())

Positive = 'Não'

y_train_act  = df[df['Partition']=='1_Training']['Contratar'] == Positive
y_test_pred  = df[df['Partition']=='2_Testing']['$XGT-Contratar'] == Positive
y_train_pred = df[df['Partition']=='1_Training']['$XGT-Contratar'] == Positive
y_test_act   = df[df['Partition']=='2_Testing']['Contratar'] == Positive

print(df[['Contratar', 'Partition']].value_counts())

def Print_Scores(pred, actual):
    import pandas as pd
    
    TP = (pred & actual).sum()
    FP = (pred & ~actual).sum()
    TN = (~pred & ~actual).sum()
    FN = (~pred & actual).sum()
    
    print('Confusion Matrix')
    Matriz = pd.DataFrame({'Positivo': [TP, FP],
                           'Negativo': [FN, TN]},
                          index=['Positivo', 'Negativo'])
    
    print(Matriz)
    
    Sensitivity = TP / (TP + FN)
    Specificity = TN / (TN + FP)
    #Precision =   TP / (TP + FP)
    
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    print('\n')
    print('Accuracy          : %.4f' % Accuracy)
    print('Balanced Accuracy : %.4f' % ((Sensitivity + Specificity)/2))
    print('\n')
    print('Specificity       : %.4f' % Specificity)
    print('Sensitivity       : %.4f' % Sensitivity)

print('Test Scores\n')
Print_Scores(y_train_pred, y_train_act)
print('\n')
print('Train Scores\n')
Print_Scores(y_test_pred, y_test_act)