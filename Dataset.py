#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fitxer de python que conté les classes Datset, Movies i Books.
@author: Lucía Rodríguez
"""

from abc import ABCMeta, abstractmethod
import numpy as np
import csv
import os
from Usuari import Usuari
from Items import Movie, Book
import pickle


class Dataset(metaclass=ABCMeta):
    """
    Classe abstracta que serveix de plantilla per a respresentar conjunt de dades (datasets)
    d’usuaris, ítems i valoracions (ratings).
    """
    
    def __init__(self, dic_usuaris={}, dic_items={}, ratings=np.empty(0)):
        self._usuaris = dic_usuaris
        self._items = dic_items
        self._ratings = ratings

    @property
    def usuaris(self):
        return self._usuaris

    @property
    def items(self):
        return self._items

    @property
    def ratings(self):
        return self._ratings

    @abstractmethod
    def llegeix(self, nom_directori: str):
        raise NotImplementedError()


class Movies(Dataset):
    """
    Classe que llegeix el document de movies i inicalitza la llista d'ítems Movies
    """
    
    def llegeix(self, nom_directori: str): 
        ruta_fitxer = os.path.join(nom_directori + 'Movies/movies.csv')
        with open(ruta_fitxer, 'r', encoding='utf8') as csv_file:
            csvreader = csv.reader(csv_file, delimiter=',')
            fields = next(csvreader)
            for index, linia in enumerate(csvreader):
                movie = Movie(index, linia[1], int(linia[0]), linia[2])  
                self._items[movie.idd] = movie  
        ruta_fitxer = os.path.join(nom_directori + 'Movies/ratings.csv')  
        with open(ruta_fitxer, 'r', encoding='utf8') as csv_file:  
            csvreader = csv.reader(csv_file, delimiter=',') 
            fields = next(csvreader)
            for linia in csvreader:  
                usuari = Usuari(int(linia[0]))  
                self._usuaris[usuari.idd] = usuari
        self._ratings = np.zeros((len(self._usuaris), len(self._items)), dtype='int8')  
        ruta_fitxer = os.path.join(nom_directori + 'Movies/ratings.csv')  
        with open(ruta_fitxer, 'r', encoding='utf8') as csv_file:  
            csvreader = csv.reader(csv_file, delimiter=',') 
            fields = next(csvreader)
            for linia in csvreader:  
                self._ratings[self._usuaris[int(linia[0])].idd - 1, self._items[int(linia[1])].ordre] = int(float(linia[2]))

       
class Books(Dataset):
    """
    Classe que llegeix el document de books i inicalitza la llista d'ítems Books
    """
    
    MAX_LLIBRES = 10000
    MAX_USUARIS = 100000

    def llegeix(self, nom_directori: str):
        ruta_fitxer = os.path.join(nom_directori + 'Books/Books.csv')  
        with open(ruta_fitxer, 'r', encoding='utf8') as csv_file:  
            csvreader = csv.reader(csv_file, delimiter=',')  
            fields = next(csvreader)
            for index, linia in enumerate(csvreader):  
                if index > self.MAX_LLIBRES:  
                    break  
                book = Book(index, linia[1], linia[0], linia[2], linia[3])  
                self._items[book.isbn] = book  
        ruta_fitxer = os.path.join(nom_directori + 'Books/Users.csv') 
        with open(ruta_fitxer, 'r', encoding='utf8') as csv_file: 
            csvreader = csv.reader(csv_file, delimiter=',') 
            fields = next(csvreader)
            for index, linia in enumerate(csvreader): 
                if index > self.MAX_USUARIS: 
                    break 
                usuari = Usuari(int(linia[0]), linia[1].split(','), linia[2])  
                self._usuaris[usuari.idd] = usuari
        self._ratings = np.zeros((len(self._usuaris), len(self._items)), dtype='int8') 
        ruta_fitxer = os.path.join(nom_directori + 'Books/Ratings.csv')  
        with open(ruta_fitxer, 'r', encoding='utf8') as csv_file:  
            csvreader = csv.reader(csv_file, delimiter=',')  
            fields = next(csvreader)
            for linia in csvreader: 
                if int(float(linia[2])) != 0:
                    if int(linia[0]) in self._usuaris and linia[1] in self._items:
                        self._ratings[int(linia[0]) - 1, self._items[linia[1]].ordre] = int(float(linia[2]))
