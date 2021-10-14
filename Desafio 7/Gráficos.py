# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 18:57:13 2020

@author: cpcle
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('leitura_iot.csv')

Linhas_ini = df.shape[0]
df.drop_duplicates(inplace=True)
Linhas_fin = df.shape[0]

print('Linhas duplicadas apagadas: %i' % (Linhas_ini - Linhas_fin))

df['Tempo'].replace({'2018-2-29' : '2018-2-28',
                           '2018-2-30' : '2018-2-18',
                           '2019-2-29' : '2019-2-28',
                           '2019-2-30' : '2019-2-28',
                           },
                          inplace=True)

df['Data'] = pd.to_datetime(df['Tempo'], format='%Y-%m-%d')
df.set_index('Data', inplace=True)

prods = ['Original_473', 'Original_269', 'Zero', 'Maçã-Verde',
         'Tangerina', 'Citrus', 'Açaí-Guaraná', 'Pêssego']

gr = df.groupby(['Data']).sum()
gr = gr[prods].groupby(pd.Grouper(freq="M")).mean()

gr.plot.line(title='Saldos Médios por Mês')
plt.legend(loc='lower left', bbox_to_anchor=(1.05, 0))
plt.show()

gr = df.groupby('Data').sum()
gr = gr[prods].groupby(pd.Grouper(freq="W")).min()
gr.plot.line(title='Saldos Médios por Semana')
plt.legend(loc='lower left', bbox_to_anchor=(1.05, 0))
plt.show()

gr = df.groupby('Data').sum()
gr = gr[prods].groupby(pd.Grouper(freq="W")).mean().rolling(window=4).mean()
gr.plot.line(title='Médias Móveis por Semana')
plt.legend(loc='lower left', bbox_to_anchor=(1.05, 0))
plt.show()

gr = df[['Estação', 'TARGET']].copy()
gr['TARGET'] = gr['TARGET'].apply(lambda x : 1 if (x == 'REABASTECER') else 0)
gr = gr.pivot_table(values='TARGET', index='Data', columns='Estação',
               aggfunc='sum', fill_value=0 )
gr = gr.groupby(pd.Grouper(freq='W')).sum()
gr.plot.line(title='Reposições por semana', figsize=(12, 9))
plt.legend(loc='lower left', bbox_to_anchor=(1, 0))
plt.savefig('graf.png')
plt.show()

 