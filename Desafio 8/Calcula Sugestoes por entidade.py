# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 10:00:45 2020

@author: cpcle
"""

from cloudant.client    import Cloudant
from cloudant.query     import Query

serviceUsername = "afddfa00-603a-477e-8cb0-08bbe6fa4cf4-bluemix"
servicePassword = "ae6025c3e11ffea8876ac30bb5c94d30462943daf1ca0b9e03921e0e5a5c2e61"
serviceURL = "https://afddfa00-603a-477e-8cb0-08bbe6fa4cf4-bluemix.cloudantnosqldb.appdomain.cloud"


def acumula(dic, key, add_value):
        if key in dic:
            dic[key] += add_value
        else:
            dic[key] =  add_value
            
        return (dic);
        

modelos = ['TORO', 'DUCATO', 'FIORINO', 'CRONOS', 'FIAT 500', 'MAREA',
           'LINEA', 'ARGO', 'RENEGADE']

resumo = {}

# connect to cloudant database
client = Cloudant(serviceUsername, servicePassword, url=serviceURL)
client.connect()

myDatabase = client['fca-entities']

# Construct a Query
query = Query(myDatabase, selector={'_id': {'$gt': 0}})

# Run query
for doc in query(limit=110)['docs']:
    
    modelo = {}
    # determina modelo
    for entity in doc['entities'].values():
        if entity['entity'] == 'MODELO' and\
           entity['mention'].upper() in modelos:
           
           modelo = acumula(modelo, entity['mention'].upper(), 1)
    
    # get model with more mentions
    if len(modelo) != 0:
        model = max(modelo, key=modelo.get)
    
        for entity in doc['entities'].values():
            resumo = acumula(resumo, (model, entity['entity']), entity['sentiment'])
        

# get the final model x entity classification
resumo = [ [x[0], x[1], resumo.get(x)]  for x in resumo.keys()]
final = {}
for entity in set([x[1] for x in resumo]):
    a = [[x[0], x[2]] for x in resumo if x[1] == entity]
    modelo = [x[0] for x in a]
    sentimento = [x[1] for x in a]
    a = dict(zip(modelo, sentimento))
    b = max(a, key=a.get); a.pop(max(a, key=a.get))    
    c = max(a, key=a.get)
    final[entity] = [b, c]
    
client.disconnect()