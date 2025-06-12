#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fitxer de python que conté la classe Usuari.
@author: Lucía Rodriguez
"""

class Usuari:
    """
    Classe que representa un usuari amb les seves característiques bàsiques
    """
    
    def __init__(self, idd='', edat=-1, poblacio='', professio=''):
        self._idd = idd
        self._edat = edat
        self._poblacio = poblacio
        self._professio = professio

    def __str__(self):
        return 'UsuariID: ' + str(self._idd) + '\nEdat: ' + str(self._edat) + '\nPoblacio: ' + self._poblacio + '\nProfessió: ' 
            + self._professio

    @property
    def idd(self):
        return self._idd

    @property
    def edat(self):
        return self._edat

    @property
    def poblacio(self):
        return self._poblacio

    @property
    def professio(self):
        return self._professio
