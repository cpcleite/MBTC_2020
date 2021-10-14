# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 17:24:31 2020

@author: cpcle
"""


from bs4 import BeautifulSoup
import requests
import json

txt_file = open('Documentos.txt')
docs = txt_file.readlines()
txt_file.close()

# Initializes file name counter
i=1

for endereco in docs:

    # acessa endereço    
    r = requests.get(endereco[0:-1])
    pagina = BeautifulSoup(r.content, 'lxml')

    # Inicializa variáveis do loop do endereço
    dic = {'author' : '',
           'body' : '',
           'title' : '',
           'type':'',
           'url':endereco[0:-1]
           }

    # URL
    #dic['url'] = endereco[0:-1]
    
    texto = []    
    
    if 'https://www.ted.com/' in endereco:    
        
        # body
        for d in pagina.find_all('div', attrs={'class':'Grid__cell flx-s:1 p-r:4'}):
            txt = [''.join(list(p.stripped_strings)) for p in d.find_all('p')]        
            texto.append(''.join(txt))
        
        dic['body'] = ''.join(texto)
        dic['type'] = 'video'
        
        for d in pagina.find_all('script', attrs={'data-spec':'q'}):
            aux = str.find(d.string, ',') + 1
            t = json.loads(d.string[aux : -1])
            dic['author'] = t['__INITIAL_DATA__']['speakers'][0]['firstname'] + ' ' +\
                t['__INITIAL_DATA__']['speakers'][0]['lastname']
            
            dic['title']=t['__INITIAL_DATA__']['name']
        
            
    elif 'https://olhardigital.com.br' in endereco:
    
        # Nome do Autor
        for d in pagina.find_all('h1', attrs={'class':'cln-nom'}):
            texto.append(''.join(d))
            
        dic['author'] = ''.join(texto)
        texto = []
        
        if dic['author'] == '':
            for d in pagina.find_all('span', attrs={'class':'meta-item meta-aut'}):
                texto.append(''.join(d))
                
            dic['author'] = ''.join(texto)                
            texto = []
            
        # Título
        for d in pagina.find_all('h1', attrs={'class':'mat-tit'}):
            texto.append(''.join(d))
            
        dic['title'] = ''.join(texto)
        texto = []

        for d in pagina.find_all('div', attrs={'class':'mat-txt'}):
            txt = [''.join(list(p.stripped_strings)) for p in d.find_all('p') if not(p.string is None)]
            texto.append(''.join(txt))
        
        dic['body'] = ''.join(texto)
        dic['type'] = 'article'
        
    else:
        
        # Nome do Autor
        for d in pagina.find_all('h4', attrs={'class': "title-single__info__author__about__name"}):
            txt = [''.join(list(p.stripped_strings)) for p in d.find_all('a') if not(p is None)]
            texto.append(''.join(txt))
        
        dic['author'] = ''.join(texto)
        texto = []
        
        # Título
        for d in pagina.find_all('h2', attrs={'class': "title-single__title__name text-white fw-600"}):
            texto.append(''.join(d))
            
        dic['title'] = ''.join(texto)
        texto = []

        for d in pagina.find_all('div', attrs={'class':'content-single__sidebar-content__content'}):
            txt = [''.join(list(p.stripped_strings)) for p in d.find_all('p') if not(p.string is None)]
            texto.append(''.join(txt))
        
        dic['body'] = ''.join(texto)
        dic['type'] = 'article'
        
    name = 'arq' + str(i) + '.json'
    with open(name, 'w+') as f:
        json.dump(dic, f)
        
    i += 1
 
        
        
    
    