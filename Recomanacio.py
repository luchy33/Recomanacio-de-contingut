#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fitxer de python que conté les classes Recomanacio, RecomanacioSimple, RecomanacioColaborativa, 
RecomanacioContingut i Avaluacio.
@author: Lucía Rodríguez
"""

from abc import ABCMeta, abstractmethod
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import math
from Usuari import Usuari
from Items import Items, Movie, Book
from Dataset import Movies, Books
from sklearn.model_selection import train_test_split


class Recomanacio(metaclass=ABCMeta):
    
    def __init__(self, dataset="", usuari=""):
        self._dataset = dataset
        self._usuari = usuari

    @property
    def usuari(self):
        return self._usuari

    @abstractmethod
    def calcul_score(self):
        raise NotImplementedError

    @staticmethod
    def preguntar_usuari(recomanacions):
        """"
        Funció utilitzada per preguntar a l'usuari quants ítems vol que se'l recomani.
        Parameters:
            - recomanacions: list[Items]. Llista amb totes les recomanacions.
        Returns:
            - None
        """
        if len(recomanacions) == 0:
            print("\nERROR: No es troba cap item per recomanar")
        else:
            res = 'si'
            while res == 'si':
                try:
                    print('\n--------------------------------------------------------------\n')
                    num_recomanacio = int(input('En total hi ha ' + str(len(recomanacions)) + ' recomanacions. Quants ítems vols que et recomanem? '))
                    print('\n--------------------------------------------------------------\n')
                    print('Les recomanacions són:\n')
                    for i in range(num_recomanacio):
                        print(recomanacions[i], '\n')
                except ValueError:  
                    print('\nERROR: El valor introduït no és un nombre enter. Torna-ho a provar.\n')
                else:
                    res = 'no'

    def mostrar_recomanacions(self, sorted_score: dict):
        """
        Funció que obté un diccionari amb puntuacions ordenades, les passa a una llista i mostra 
        per pantallas tantes recomanacions com l'usuari vulgui.
        Parameters:
            - sorted_score: dict[float: int]. Diccionari on la clau és la puntuació que obté 
                            l'ítem i el valor és l'ordre d'aquest.
        Returns:
            - recomanacions: list[Items]. Llista amb totes les recomanacions.
        """
        recomanacions = []
        ordres = {item.ordre: item for item in self._dataset.items.values()}
        for puntuacio, llista in sorted_score:
            for ordre in llista:
                item = ordres.get(ordre)
                if item:
                    recomanacions.append(item)
                else:
                    pass
        return recomanacions

    def avalua(self, llindar: int, n: int):
        """
        Funció per obtenir les valoracions d'un usuari específic sobre ítems.
        Parameters:
            - llindar: int. Llindar per filtrar valoracions
            - n: int. Nombre de valoracions
        Returns:
            - valoracio: array. Totes les valoracions qu3e s'han fet
        """
        valoracio = self._dataset.ratings[self._usuari.idd - 1, :]  
        valoracio_no_zero = valoracio != 0  
        ordres = {item.ordre: item for item in self._dataset.items.values()}
        x = 0
        if any(valoracio_no_zero):
            res = 'si'
        else:
            print("L'usuari no ha valorat res.")
            return False
        print("\nLes valoracions de l'usuari són:")
        for i in range(len(valoracio)):
            if (x < n):
                if valoracio_no_zero[i] != False and valoracio[
                    i] > llindar:  
                    item = ordres[i]  
                    print("Id: ", item.title, 'Valoracio: ', valoracio[i])
                    x += 1
        return valoracio[valoracio_no_zero]


class RecomanacioSimple(Recomanacio):
    
    def __init__(self, dataset='', usuari='', min_vots=-1, array_avg=''):
        super().__init__(dataset, usuari)
        self._min_vots = min_vots
        self._array_avg = array_avg

    def calcul_avg(self):
        """
        Funció que crea una matriu amb les mitjanes de les puntuacions de cada item, indica els
        vots de cada item i calcula la mitjana de les mitjanes de les puntuacions dels items.
        Parameters:
            - None
        Returns:
            - avg_array : np.array. Matriu de numpy on la primera fila representa les puntuacions 
                          mitjanes de cada item, cada columna un item i la segona fila representa 
                          el número de vots per cada item. 
            - avg_global: float. Mitjana de la primera fila de la matriu avg_array
        """
        avg_array = np.zeros((2, len(self._dataset.items)), dtype=np.float32)
        ratings = self._dataset.ratings
        user_idx = self._usuari.idd - 1
        for item in range(ratings.shape[1]): 
            nota = ratings[:, item]
            notes = np.delete(nota, user_idx)
            valors_no_zero = notes[notes != 0]
            n_valors = len(valors_no_zero)
            if n_valors >= self._min_vots: 
                avg_array[0, item] = valors_no_zero.mean()
                avg_array[1, item] = n_valors
        puntuacions = avg_array[0][avg_array[0] != 0]
        if len(puntuacions) > 0:
            avg_global = np.nanmean(puntuacions)
        else:
            avg_global = 0.0
        return avg_array, avg_global

    def calcul_score(self):
        """
        Funció que calcula les puntuacions per als items que l'usuari al que recomanem no ha vist i 
        mostra per pantalla els num_recomanacio items que l'usuari desitji.
        Parameters:
            - None
        Returns:
            - dic_score: list[Tuple[float, int]]. La primera posició de la tupla és la puntuació i la 
                         segona l'ordre en l'array de ratings.
            - recomanacions: list[Items]. Llista amb tots els items que recomanaríem en ordre decreixent.
        """
        avg_array, avg_global = self.calcul_avg()
        dic_score = {}
        linia_v1 = self._dataset.ratings[self._usuari.idd - 1, :]
        items_zero = np.where(linia_v1 == 0)[0]
        items_valids = [i for i in items_zero if avg_array[1, i] >= self._min_vots]
        for columna in items_valids:
            avg_item = avg_array[0, columna]
            num_vots = avg_array[1, columna]
            denom = num_vots + self._min_vots
            if denom == 0:
                score = 0
            else:
                score = ((num_vots * avg_item) / denom) + ((self._min_vots * avg_global) / denom)
            if score in dic_score:
                dic_score[score].append(columna)
            else:
                dic_score[score] = [columna]
        sorted_score = sorted(dic_score.items(), reverse=True)
        recomanacions = self.mostrar_recomanacions(sorted_score)
        return dic_score, recomanacions

    def avalua(self, llindar: int, n: int):
        valoracio_usr = super().avalua(llindar, n)
        if type(valoracio_usr) == bool:
            print("Com l'usuari no ha puntuat, no podem avaluar")
            return False
        prediccio_dic, recomanacions = RecomanacioSimple(self._dataset, self._usuari, self._min_vots).calcul_score()
        prediccions = np.array([prediccio_dic.keys()])
        puntuacions = {}
        for puntuacio, ordre in prediccio_dic.items():
            for element in ordre:
                puntuacions[element] = puntuacio
        print("\nLes prediccions de l'usuari són:")
        for i in range(n):
            ordre = recomanacions[i].ordre
            print("Id: ", recomanacions[i].title, "Puntuació: ", puntuacions[ordre])
        if valoracio_usr.size != 0: 
            avaluem = Avaluacio(self._dataset, self._usuari, llindar, prediccions, valoracio_usr, n)
            avaluem.mesures_comparacio()
        else:
            print("L'usuari no ha puntuat, per tant no podem avaluar")


class RecomanacioColaborativa(Recomanacio):
    
    def __init__(self, dataset='', usuari='', k=-1):
        super().__init__(dataset, usuari)
        self._k = k

    def similitud_vectors(self):
        """
        Funció que retorna la similitud entre dos usuaris.
        Parameters:
            - None
        Returns:
            - dic_similituds: dict {distancia: fila}. Diccionari on la clau és la distància
                             (float) entre dos usuaris i el valor la fila de la matriu (idd-1 
                              del usuari al que hem calculat la distància respecte a l'usuari
                              al que recomanem items).
        """
        dic_similituds = {}
        v1 = self._usuari 
        puntuacions_v1 = self._dataset.ratings[v1.idd - 1, :]  
        for fila in range(
                len(self._dataset.ratings)):
            if fila != v1.idd - 1:
                puntuacions_v2 = self._dataset.ratings[fila, :] 
                numerador = 0
                denominador_v1 = 0
                denominador_v2 = 0
                for element_v1, element_v2 in zip(puntuacions_v1, puntuacions_v2):
                    if element_v1 != 0 and element_v2 != 0:  
                        numerador += element_v1 * element_v2
                        denominador_v1 += element_v1 ** 2
                        denominador_v2 += element_v2 ** 2
                if denominador_v1 == 0 or denominador_v2 == 0:  
                    distancia = None  
                else:
                    distancia = numerador / ((denominador_v1 ** 0.5) * (denominador_v2 ** 0.5))
                    dic_similituds[distancia] = fila
        return dic_similituds

    def calcul_k_similars(self):
        """
        Funció que crea un nou array afegint dues columnes: similitud entre usuaris i la mitjana 
        de les puntuacions de cada usuari. Ho afegim a l'array principal ja que si en fem un de 
        nou no sabrem quina posició té i no podrem buscar-lo després.
        Parameters:
            - None
        Returns:
            - array_final: np.array. Array de ratings on s'han afegit dues columnes. Sol els k 
                           elements tindran valors en aquestes posicions, la resta el valor -1.
        """
        dic_similituds = self.similitud_vectors()
        similituds_sorted = sorted(dic_similituds.items())[::-1]
        similituds = np.ones(shape=len(self._dataset.ratings), dtype="float64") * (-1)
        for i in range(self._k):
            similitud, ordre = similituds_sorted[i][0], similituds_sorted[i][1]  
            if similitud != -1:  
                similituds[ordre] = similitud 
            else: 
                print("No hi ha k pel·lícules que et poguem recomanar")
        array = np.concatenate((self._dataset.ratings, similituds.reshape(-1, 1)), axis=1) 
        mitjanes = np.ones(shape=len(self._dataset.ratings), dtype="float64") * (-1)
        for usuari in range(len(self._dataset.ratings)):
            if usuari == self._usuari.idd - 1:  
                mitjana = self._dataset.ratings[usuari, :][(self._dataset.ratings[usuari,:]) != 0].mean()  
                mitjanes[usuari] = mitjana 
            elif array[usuari][-1] != -1:
                mitjana = self._dataset.ratings[usuari, :][(self._dataset.ratings[usuari, :]) != 0].mean()
                mitjanes[usuari] = mitjana
        array_final = np.concatenate((array, mitjanes.reshape(-1, 1)), axis=1)
        return array_final

    def calcul_score(self):
        """
        Funció que retorna una llista amb els k elements recomanats, ordenats de major a menor i 
        mostra per pantalla tants com vulgui l'usuari.
        Parameters:
            - None
        Returns:
            - dic_score: list[Tuple[float, int]]. La primera posició de la tupla és la puntuació 
                         i la segona l'ordre en l'array de ratings.
            - recomanacions: list[Items]. Llista amb tots els ítems recomanats per l'Usuari, 
                             ordenats de major a menor probabilitat de que li agradi.
        """
        dic_score = {}  
        array = self.calcul_k_similars()
        linia_v1 = array[self._usuari.idd - 1, :]
        items = np.where(linia_v1[:-2] == 0)  
        usuaris = np.where(array[:, -2] != -1)
        for columna in items[0]:
            numerador = 0
            denominador = 0
            for fila in usuaris[0]:  
                numerador += array[fila, -2] * (array[fila, columna] - array[fila, -1]) 
                denominador += array[fila, -2]
            puntuacio = linia_v1[-1] + (numerador / denominador)
            if puntuacio in dic_score:
                dic_score[puntuacio].append(columna)  
            else:
                dic_score[puntuacio] = [columna]  
        sorted_score = (sorted(dic_score.items()))[::-1]  
        recomanacions = self.mostrar_recomanacions(sorted_score)
        return dic_score, recomanacions

    def avalua(self, llindar: int, n: int):
        valoracio_usr = super().avalua(llindar, n)
        if type(valoracio_usr) == bool:
            return False
        prediccio_dic, recomanacions = RecomanacioColaborativa(self._dataset, self._usuari, self._k).calcul_score()
        prediccions = np.array([prediccio_dic.keys()])
        puntuacions = {element: puntuacio for puntuacio, ordre in prediccio_dic.items() for element in ordre}
        print("\nLes prediccions de l'usuari són:")
        for i in range(n):
            ordre = recomanacions[i].ordre
            print("Id: ", recomanacions[i].title, "Puntuació: ", puntuacions[ordre])
        if valoracio_usr.size != 0 or valoracio_usr[0] is not np.isnan():
            avaluem = Avaluacio(self._dataset, self._usuari, llindar, prediccions, valoracio_usr, n) 
            avaluem.mesures_comparacio()
        else:
            print("L'usuari no ha puntuat, per tant no podem avaluar")
            

class RecomanacioContingut(Recomanacio):
    
    def __init__(self,  tipus='movies', dataset='', usuari='', pmax=-1):
        super().__init__(dataset, usuari)
        self._pmax = pmax
        self._tipus = tipus

    @property
    def pmax(self):
        return self._pmax

    def delete_words(self, nom_directori: str):
        """
        Funció que elimina els articles i paraules no rellevants d'un string.
        Parameters:
            - nom_directori: str. Nom del directori on es troben els fixterrs amb les paraules 
                             no rellevants.
        Returns:
            - invalid: List. Llista de str irellevants de un str (caràcters invalids).
        """
        ruta_fitxer_prep = os.path.join(nom_directori + '/prepositions.csv') 
        ruta_fitxer_num = os.path.join(nom_directori + '/numeros.txt')
        invalid = ['!', '?', '¿', '\)', '\(', '/', '$', '&', '%', '#', '@', ',', ':', '.', '\"', '\'', '<', '>', '*'] 
        with open(ruta_fitxer_prep, 'r') as Invalid:
            for linia in Invalid:
                word = linia[:-1].split(',')
                invalid.append(word[1])
        with open(ruta_fitxer_num, 'r') as Invalid:
            text = Invalid.read().split(',')
            for i in text:
                invalid.append(i)
        return invalid

    def calcul_tf_idf(self):
        """
        Funció que retorna una matriu amb la representació tf-idf.
        Parameters:
            - None
        Returns:
            - None
        """
        if self._tipus == 'movies':
            ll_generes = [item.genere for item in self._dataset.items.values()] 
            tfidf = TfidfVectorizer(stop_words='english')
            return tfidf.fit_transform(ll_generes).toarray()
        elif self._tipus == 'books':
            invalid = self.delete_words('Datasets')
            ll_generes = [item.title + '|' + item.autor for item in self._dataset.items.values()]
            tfidf = TfidfVectorizer(stop_words=invalid)
            return tfidf.fit_transform(ll_generes).toarray()
        else:
            raise ValueError("Tipus de contingut no suportat: només 'movie' o 'book'")

    def perfil_usuari(self, tdidf_matriu):
        """
        Funció que retorna el perfil d'un usuari.
        Parameters:
            - tdidf_matriu: matriu tfidf
        Returns:
            - perfil_usuari: np.array. Vector que correspon al perfil d'un usuari.
        """
        puntuacio_u1 = self._dataset.ratings[self._usuari.idd - 1, :] 
        if np.sum(puntuacio_u1) == float(0):
            print('L\'usuari no ha puntuat cap ítem. Per tant, no realitzarem cap recomanació.')
            return -1
        else:
            perfil_usuari = np.sum(np.multiply(puntuacio_u1, tdidf_matriu.T), axis=1) / np.sum(puntuacio_u1)
            return perfil_usuari

    def calcul_distancia(self, tdidf_matriu):
        """
        Funció que calcula la distància entre l'usuari i tots els ítems.
        Parameters:
            - tdidf_matriu: matriu tfidf
        Returns:
            - distancia_matriu: array numpy
        """
        perfil_usuari = self.perfil_usuari(tdidf_matriu) 
        if type(perfil_usuari) == int: 
            return -1
        else:  
            distancia_matriu = np.dot(tdidf_matriu, perfil_usuari)
            return distancia_matriu

    def calcul_score(self):
        """
        Funció que calcula la puntuació final de cada ítem.
        Parameters:
            - None
        Returns:
            - puntuacions: list[Tuple[float, int]]. Llista de tuples. La primera posició de 
                         la tupla és la puntuació i la segona l'ordre en l'array de ratings.
            - recomanacions: list[Items]. Llista on es retornen tots els ítems recomanats per 
                            l'Usuari, ordenats de major a menor probabilitat de que li agradi.
        """
        matriu_tf_idf = self.calcul_tf_idf()
        distancia_matriu = self.calcul_distancia(matriu_tf_idf)
        if type(distancia_matriu) != int:
            puntuacio_final = (self._pmax * distancia_matriu)
            puntuacions = {puntuacio: [i] for i, puntuacio in enumerate(puntuacio_final)}
            sorted_score = (sorted(puntuacions.items()))[::-1]
            recomanacions = self.mostrar_recomanacions(sorted_score)
        return puntuacions, recomanacions

    def avalua(self, llindar: int, n: int):
        valoracio_usr = super().avalua(llindar, n)
        if type(valoracio_usr) == bool:
            print("Com l'usuari no ha puntuat,no podem avaluar")
        if valoracio_usr.size != 0:
            avaluem = Avaluacio(self._dataset, self._usuari, llindar, n)
            avaluem.calculem_prediccions_contingut()
            avaluem.mesures_comparacio()
        else:
            print("L'usuari no ha puntuat, per tant no podem avaluar")


class Avaluacio:
    
    def __init__(self, dataset='', usuari='', llindar=-1, prediccio=np.zeros(1), valoracions=np.zeros(1), n=-1):
        self._dataset = dataset
        self._usuari = usuari 
        self._prediccions = prediccio  
        self._valoracions = valoracions
        self._llindar = llindar
        self._n = n

    def crear_conjunt_train_test(self, llindar):
        train, test = train_test_split(self._dataset.ratings, test_size=0.2, random_state=50)
        self._llindar = llindar
        return train, test

    def entrenem_dades(self):
        """
        Funció que prepara les dades d’entrenament per al sistema de recomanació basat en contingut 
        i genera un perfil de l’usuari (basat en les seves valoracions) i la matriu TF-IDF dels ítems.
        """
        self._valoracions = self._dataset.ratings[self._usuari.idd - 1, :][
            self._dataset.ratings[self._usuari.idd - 1, :] != 0]
        if self._valoracions.size != 0:  
            train, test = self.crear_conjunt_train_test(self._llindar)
        else:
            return None
        self._dataset._ratings = train
        if self._llindar == 4:
            tf_idf = RecomanacioContingut("movies", self._dataset, self._usuari).calcul_tf_idf()
            perfil = RecomanacioContingut("movies", self._dataset, self._usuari).perfil_usuari(tf_idf)
        else: 
            tf_idf = RecomanacioContingut("books", self._dataset, self._usuari).calcul_tf_idf()
            perfil = RecomanacioContingut("books", self._dataset, self._usuari).perfil_usuari(tf_idf)
        return perfil, tf_idf

    def calculem_prediccions_contingut(self):
        """
        Funció que calcula les prediccions de les puntuacions finals de cada ítem
        """
        train, test = self.crear_conjunt_train_test(self._llindar)
        perfil_usuari, matriu_tf_idf = self.entrenem_dades()
        self._dataset._ratings = test  
        if self._llindar == 4: 
            recomanacio = RecomanacioContingut()
            recomanacio._pmax = 5
        else:
            recomanacio = RecomanacioContingut()
            recomanacio._pmax = 10
        if type(perfil_usuari) == int:
            return -1  
        else:
            distancia_matriu = np.dot(matriu_tf_idf, perfil_usuari)  
        if type(distancia_matriu) != int:
            puntuacio_final = (recomanacio.pmax * distancia_matriu)
            puntuacions_ordre = {puntuacio: [i] for i, puntuacio in enumerate(puntuacio_final)}
            puntuacions = [puntuacio for i, puntuacio in enumerate(puntuacio_final)]
        return self._prediccions, np.array([puntuacions])

    def mesures_comparacio(self):
        """
        Funció que calcula dues mètriques d’avaluació (MAE i RMSE) entre les valoracions reals 
        d’un usuari i les prediccions fetes pel sistema de recomanació:
        """
        valoracio_list = self._dataset.ratings[self._usuari.idd - 1, :][
            self._dataset.ratings[self._usuari.idd - 1, :] != 0].tolist()
        pred_list = [x for i in self._prediccions for x in i]
        numerador = [abs(prediccio - valoracio) for prediccio, valoracio in zip(pred_list, valoracio_list) if (valoracio != 0)]
        sum_numerador = np.sum(numerador)
        length_numerador = len(numerador)
        if length_numerador > 0 and not np.isnan(sum_numerador):
            MAE = sum_numerador / length_numerador
        else:
            MAE = None
        combinades = [(v, p) for v, p in zip(valoracio_list, pred_list) if v != 0]
        if len(combinades) > 0:
            dif_quad = [(v - p) ** 2 for v, p in combinades]
            RMSE = math.sqrt(sum(dif_quad) / len(dif_quad))
        else:
            RMSE = None
        print('\nEl resultat de l\'avaluacio és:\n\tMAE: ', MAE, '\n\tRMSE:', RMSE)
        