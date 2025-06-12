#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fitxer de python que conté la classe Principal (collector) i la funció principal del projecte: main().
@author: Lucía Rodríguez
"""

import numpy as np
import pickle
from Usuari import Usuari
from Dataset import Movies, Books
from Recomanacio import RecomanacioColaborativa, RecomanacioSimple, RecomanacioContingut

# CONTROLLER: Classe Principal
class Principal:
    
    ITEMS = ['Pel·lícules', 'Llibres']
    SISTEMES = ['Simple', 'Col·laboratiu', 'Contingut']
    NOM_DIRECTORI = 'Datasets/'
    LLINDAR_MOVIES = 4
    LLINDAR_BOOKS = 7

    def __init__(self):
        self._dataset = ""
        self._sistema = ""
        self._usuari = ""
        self._tipus_dataset = ""
        self._llindar = -1

    @property
    def usuari(self):
        return self._usuari

    def inicialitza(self):
        print('Quina base de dades vols executar?  \n\t1. Pel·lícules\n\t2. Llibres\n')
        res = 'si'
        while res == 'si':
            try:
                n_dataset = int(input(' Opció (1-2): '))
                if n_dataset == 1 or n_dataset == 2:  
                    items = self.ITEMS[n_dataset - 1]  #items es inutil
                    self._tipus_dataset = items
                    print('\n--------------------------------------------------------------\n')
                    res = 'no'
                else:
                    raise ValueError
            except ValueError:
                print('ERROR: El valor introduït no és un nombre vàlid. Torna-ho a provar!\n')
        print('Com vols inicialitzar el dataset?  \n\t1. Llegint els fitxers\n\t2. Important l\'arxiu pickle\n')
        res = 'si'
        while res == 'si':
            try:
                inicialitzacio = int(input(' Opció (1-2): '))
                if inicialitzacio == 1 or inicialitzacio == 2:
                    res = 'no'
                else:
                    raise ValueError
            except ValueError:
                print('ERROR: El valor introduït no és un nombre vàlid. Torna-ho a provar!\n')
        if inicialitzacio == 1:
            if n_dataset == 1:
                self._dataset = Movies()  
                self._dataset.llegeix(self.NOM_DIRECTORI)
                self._llindar = self.LLINDAR_MOVIES
            else:  
                self._dataset = Books()  
                self._dataset.llegeix(self.NOM_DIRECTORI)  
                self._llindar = self.LLINDAR_BOOKS
        else:  
            if n_dataset == 1: 
                with open("Movies.dat", 'rb') as fitxer:  
                    self._dataset = pickle.load(fitxer)  
            else:  
                with open("Books.dat", 'rb') as fitxer:
                    self._dataset = pickle.load(fitxer)
        res = 'si'
        while res == 'si':
            try:
                print('\n--------------------------------------------------------------\n')
                print('Introdueix a quin usuari vols recomanar-li items (per finalitzar l\'execució fica un espai en blanc).')
                usuari = input('\nUsuari (1-' + str(len(self._dataset.usuaris)) + '): ')  
                if 1 <= int(usuari) <= len(self._dataset.usuaris):
                    res = 'no'  
                    self._usuari = Usuari(int(usuari))
                elif 1 > int(usuari) > self._dataset.usuaris:
                    raise TypeError
            except TypeError:
                print("\nERROR: El nombre introduït no és dins l'interval. Torna-ho a provar.\n")
            except ValueError:
                if usuari in ' \n':
                    res = 'no'
                    self._usuari = ' \n'
                    print('\nTancant programa...')
                else:
                    print("\nERROR: El nombre introduït ha de ser enter. Torna-ho a provar.\n")
        if self._usuari != ' \n':
            print('\n--------------------------------------------------------------\n')
            print('Mètodes de recomanació disponibles:\n\t1. Simple\n\t2. Col·laboratiu\n\t3. Basat en contingut')
            res = 'si'
            while res == 'si':
                try:
                    metode = int(input('\n Opció (1-3): '))
                    if 3 >= metode >= 1:
                        sistema = self.SISTEMES[metode - 1]
                        res = 'no'
                    else:
                        raise ValueError
                except ValueError: 
                    print('ERROR: El valor introduït no és un nombre vàlid. Torna-ho a provar.\n')

            if self._usuari != ' \n':  
                if sistema == self.SISTEMES[0]: 
                    res = 'si'  
                    while res == 'si':
                        try:
                            print('\n--------------------------------------------------------------\n')
                            minim_vots = int(input('Introdueix els vots mínims: '))
                        except ValueError:
                            print('\nERROR: El valor introduït no és un nombre enter. Torna-ho a provar.\n')  
                        else:  
                            res = 'no'  
                    self._sistema = RecomanacioSimple(self._dataset, self._usuari, minim_vots)
                elif sistema == self.SISTEMES[1]:
                    res = 'si'  
                    while res == 'si':
                        try:
                            print('\n--------------------------------------------------------------\n')
                            k = int(input("Introdueix el nombre d'usuaris més similars a tu (k): "))
                        except ValueError:
                            print( "\nERROR: El valor introduït no és un nombre enter. Torna-ho a provar.\n")  
                        else: 
                            res = 'no' 
                    self._sistema = RecomanacioColaborativa(self._dataset, self._usuari, k)

                elif sistema == self.SISTEMES[2]:
                    if self._tipus_dataset == self.ITEMS[0]:
                        self._sistema = RecomanacioContingut("movies", self._dataset, self._usuari, 5)
                    else:
                        self._sistema = RecomanacioContingut("books",self._dataset, self._usuari, 10)

    def recomana(self):
        dic, recomanacions = self._sistema.calcul_score()
        self._sistema.preguntar_usuari(recomanacions)  

    def avalua(self):
        print('\n--------------------------------------------------------------\n')
        n = int(input("Quins N millors ítems vols considerar? "))
        self._sistema.avalua(self._llindar, n)  

    def executa(self):
        print('\n--------------------------------------------------------------\n')
        print('Quina acció vols realitzar?  \n\t1. Recomanar', self._tipus_dataset.lower(),'\n\t2. Avaluar les recomanacions fetes\n')
        res = 'si'
        while res == 'si':
            try:
                accio = int(input(' Opció (1-2): '))
                if accio == 1 or accio == 2:  
                    res = 'no'  
            except ValueError:
                print('ERROR: El valor introduït no és un nombre vàlid. Torna-ho a provar!\n')
        if accio == 1:  
            self.recomana()  
        else:
            self.avalua() 

# FUNCIÓ PRINCIPAL
def main():
    print('\nBenvingut/da al sistema de recomanació!\n')
    principal = Principal()
    principal.inicialitza()
    while principal.usuari != ' \n':
        principal.executa()
        print('\nHa finalitzat el sistema de recomanació!')
        print("Tornem a començar...\n")
        principal.inicialitza()
    return ("\nEl programa ha acabat.")


res = main()
print(res)
