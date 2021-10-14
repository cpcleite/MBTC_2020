# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 12:03:45 2020

@author: cpcle
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('leitura_iot.csv')

Linhas_ini = df.shape[0]
df.drop_duplicates(inplace=True)
Linhas_fin = df.shape[0]

print('Linhas duplicadas apagadas: %i' % (Linhas_ini - Linhas_fin))

prods = ['Original_473', 'Original_269', 'Zero', 'Maçã-Verde',
         'Tangerina', 'Citrus', 'Açaí-Guaraná', 'Pêssego']

df['Tempo'].replace({'2018-2-29' : '2018-2-28',
                           '2018-2-30' : '2018-2-18',
                           '2019-2-29' : '2019-2-28',
                           '2019-2-30' : '2019-2-28',
                           },
                          inplace=True)

df['Data'] = pd.to_datetime(df['Tempo'], format='%Y-%m-%d')
df.set_index('Data', inplace=True)


gr = df.copy() #df[df['Estação']=='Luz']

a = gr[prods].apply(lambda x : (x - np.mean(x))/np.std(x), axis=0)
a['Minimo'] = a[prods].apply(lambda x: np.min(x[prods]), axis=1)
a = pd.concat([a, gr['TARGET']], axis=1)
a = a.sort_values('Data').reset_index(drop=False)

a = a.pivot_table(index='Data', columns='TARGET', values='Minimo')


fig, ax = plt.subplots()
ax.scatter(x=a.index,
           y=a['NORMAL'],
           c='blue',
           label='Normal',
           alpha=0.3
           )
ax.scatter(x=a.index,
           y=a['REABASTECER'],
           c='red',
           label='Reabastecer',
           alpha=0.3
           )

ax.set_title('Reposição')
ax.legend()
plt.show()

Estacoes = df['Estação'].value_counts()

gr = df.copy() #df[df['Estação']=='Luz']

a = gr[prods].apply(lambda x : (x - np.mean(x))/np.std(x), axis=0)
a['Minimo'] = a[prods].apply(lambda x: np.min(x[prods]), axis=1)
a = pd.concat([a, gr['TARGET']], axis=1)
a = a.sort_values('Data').reset_index(drop=False)

a = a.pivot_table(index='Data', columns='TARGET', values='Minimo')

c=0; l=0
for est in Estacoes[:9]:
    a = df[df['Estação']==est]
    ax[l, c].scatter(x=a.index,
                     y=a['NORMAL'],
                     c='blue',
                     label='Normal',
                     alpha=0.3
                     )
    ax[l, c].scatter(x=a.index,
                     y=a['REABASTECER'],
                     c='blue',
                     label='Reabastecer',
                     alpha=0.3
                     )
    ax[l,c].set_title(est)
    
    if c==2:
        c=0
        l+=1
    else:
        c+=1

    
    


