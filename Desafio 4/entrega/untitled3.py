# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 13:05:28 2020

@author: cpcle
"""

import pandas as pd

df = pd.read_csv('results.csv')

df.to_excel('res.xlsx')