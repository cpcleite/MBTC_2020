# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 10:00:45 2020

@author: cpcle
"""
from os import listdir
from os.path import isfile, join

import re

from cloudant.client import Cloudant
from cloudant.error import CloudantException
from cloudant.result import Result, ResultByKey

serviceUsername = "afddfa00-603a-477e-8cb0-08bbe6fa4cf4-bluemix"
servicePassword = "ae6025c3e11ffea8876ac30bb5c94d30462943daf1ca0b9e03921e0e5a5c2e61"
serviceURL = "https://afddfa00-603a-477e-8cb0-08bbe6fa4cf4-bluemix.cloudantnosqldb.appdomain.cloud"

client = Cloudant(serviceUsername, servicePassword, url=serviceURL)
client.connect()

myDatabase = client['fca']

mypath = r'C:\Users\cpcle\OneDrive\Documentos\Celso\Maratona Behind the Code 2020\Desafio 8'

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) 
                                        and re.search('^train_[0-9]*\.txt$' ,f)]

# Create documents using the sample data.
# Go through each row in the array
for document in onlyfiles:

    with open(join(mypath, document), encoding='utf-8') as f:
         texto = f.read()
         
    # Create a JSON document
    jsonDocument = {
         "arquivo": document,
         "texto": texto
    }

    # Create a document using the Database API.
    newDocument = myDatabase.create_document(jsonDocument)

    
client.disconnect()