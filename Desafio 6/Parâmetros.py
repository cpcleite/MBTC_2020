# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 12:49:20 2020

@author: cpcle
"""
#### Parametros XGB
model = xgb.XGBClassifier(booster='gbtree')

parametros = { 'model__n_estimators'     : [100, 200],
               'model__eta'              : [0.03, 0.05, 0.10],
               'model__max_depth'        : [6],
               'model__min_child_weight' : [2],
               'model__gamma'            : [0.5],
               'model__subsample'        : [1],
               'model__colsample_bytree' : [1],
               'model__objective'        : ['multi:softmax'],
               'model__eval_metric'      : ['merror'],
               'model__num_class'        : [6]
    }