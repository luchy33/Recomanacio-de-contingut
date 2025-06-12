#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fitxer de python que conté les classes Items, Movie i Book.
@author: Lucía Rodríguez
"""

from abc import ABCMeta, abstractmethod


class Items(metaclass=ABCMeta):
    """
    Classe abstracta que representa un ítem genèric amb un ordre i un títol
    """
    
    def __init__(self, ordre=-1, title=''):
        self._ordre = ordre
        self._title = title
    
    @abstractmethod
    def __str__(self):
        raise NotImplementedError()
    
    @property
    def ordre(self):
       return self._ordre
   
    @property
    def title(self):
       return self._title
   
    
class Movie(Items):
    """
    Representa un ítem de tipus pel·lícula (Movie)
    """
    
    def __init__(self, ordre=-1, title='', idd='', genere=''):
        super().__init__(ordre, title)
        self._idd = idd
        self._genere = genere
        
    def __str__(self):
        return 'MovieId: ' + str(self._idd) + '\nTitle: ' + self._title + '\nGènere/s: ' + self._genere
    
    @property
    def idd(self):
        return self._idd
    
    @property
    def genere(self):
        return self._genere

    
class Book(Items):
    """
    Representa un ítem de tipus llibre (Book)
    """
    
    def __init__(self,  ordre=-1, title='', isbn ='', autor='', any_publicacio=-1):
        super().__init__(ordre, title)
        self._isbn = isbn
        self._autor = autor
        self._any_publicacio = any_publicacio
    
    def __str__(self):
        return 'Book ISBN: ' + str(self._isbn) + '\nTitle: ' + self._title + '\nAutor: ' + self._autor + '\nAny publicacio:' 
                + str(self._any_publicacio)
        
    @property
    def isbn(self):
        return self._isbn
        
    @property
    def autor(self):
        return self._autor
    
    @property
    def any_publicacio(self):
        return self._any_publicacio
