# Inteligencia Artificial para la Ciencia de los Datos
# Implementacion de clasificadores 
# Dpto. de C. de la Computacion e I.A. (Univ. de Sevilla)
# ===================================================================


# --------------------------------------------------------------------------
# Autores del trabajo:
#
# APELLIDOS: Manuel
# NOMBRE: Fernandez
#
#
# APELLIDOS: Miguel Angel
# NOMBRE: Perez Cutino
# TODO ->>> Cambiar tildes y n en los nombres. Puesto asi porque son 
# caracteres raros y Vscode en Linux me esta dando el berro
# ----------------------------------------------------------------------------


# *****************************************************************************
# HONESTIDAD ACADEMICA Y COPIAS: un trabajo practico es un examen, por lo
# que debe realizarse de manera individual. La discusion y el intercambio de
# informacion de caracter general con los companeros se permite, pero NO AL
# NIVEL DE CODIGO. Igualmente el remitir codigo de terceros, OBTENIDO A TRAVES
# DE LA RED o cualquier otro medio, se considerara plagio. Si tienen
# dificultades para realizar el ejercicio, consulten con el profesor. En caso
# de detectarse plagio, supondra una calificacion de cero en la asignatura,
# para todos los alumnos involucrados. Sin perjuicio de las medidas
# disciplinarias que se pudieran tomar. 
# *****************************************************************************


# IMPORTANTE: NO CAMBIAR EL NOMBRE NI A ESTE ARCHIVO NI A LAS CLASES, METODOS
# Y ATRIBUTOS QUE SE PIDEN. EN PARCTICULAR: NO HACERLO EN UN NOTEBOOK.

# NOTAS: 
# * En este trabajo NO se permite usar Scikit Learn, excepto algunos metodos 
#   que se indican expresamente en el enunciado. En particular no se permite
#   usar ningun clasificador de Scikit Learn.  
# * Se permite (y se recomienda) usar numpy.  


# ====================================================
# PARTE I: IMPLEMENTACION DEL CLASIFICADOR NAIVE BAYES
# ====================================================

# Se pide implementar el clasificador Naive Bayes, en su version categorica
# con suavizado y log probabilidades (descrito en el tema 2, diapositivas 31 a
# 42). En concreto:


# ----------------------------------
# I.1) Implementacion de Naive Bayes
# ----------------------------------

# Definir una clase NaiveBayes con la siguiente estructura:

# class NaiveBayes():

#     def __init__(self,k=1):
#                 
#          .....
         
#     def entrena(self,X,y):

#         ......

#     def clasifica_prob(self,ejemplo):

#         ......

#     def clasifica(self,ejemplo):

#         ......


# * El constructor recibe como argumento la constante k de suavizado (por
#   defecto 1) 
# * Metodo entrena, recibe como argumentos dos arrays de numpy, X e y, con los
#   datos y los valores de clasificacion respectivamente. Tiene como efecto el
#   entrenamiento del modelo sobre los datos que se proporcionan.  
# * Metodo clasifica_prob: recibe un ejemplo (en forma de array de numpy) y
#   devuelve una distribucion de probabilidades (en forma de diccionario) que
#   a cada clase le asigna la probabilidad que el modelo predice de que el
#   ejemplo pertenezca a esa clase. 
# * Metodo clasifica: recibe un ejemplo (en forma de array de numpy) y
#   devuelve la clase que el modelo predice para ese ejemplo.   

# Si se llama a los metodos de clasificacion antes de entrenar el modelo, se
# debe devolver (con raise) una excepcion:

from math import e
import numpy as np
import random

from auxiliars.naive_bayes import NaiveBayes
from data.jugar_tenis import X_tenis, y_tenis

# Ejemplo "jugar al tenis":

print("\n ++ Play Tenis with Naive Bayes ++ \n")

nb_tenis=NaiveBayes(k=0.5)
nb_tenis.entrena(X_tenis,y_tenis)
ej_tenis = np.array(['Soleado','Baja','Alta','Fuerte'])
print(nb_tenis.clasifica_prob(ej_tenis))
# {'no': 0.7564841498559081, 'si': 0.24351585014409202}
print(nb_tenis.clasifica(ej_tenis))
# 'no'

# ----------------------------------------------
# I.2) Implementacion del calculo de rendimiento
# ----------------------------------------------

# Definir una funcion "rendimiento(clasificador,X,y)" que devuelve la
# proporcion de ejemplos bien clasificados (accuracy) que obtiene el
# clasificador sobre un conjunto de ejemplos X con clasificacion esperada y. 

def rendimiento(clasificador, X, y, sklearn_clf=False):
    preds = list(map(clasificador.clasifica, X)) if not sklearn_clf else clasificador.predict(X)

    preds = np.array(preds)
    y = np.array(y)

    acc = (preds == y).sum()
    return acc/len(y)

# Ejemplo:

print(rendimiento(nb_tenis,X_tenis,y_tenis))
# 0.9285714285714286

# --------------------------
# I.3) Aplicando Naive Bayes
# --------------------------

# Usando el clasificador implementado, obtener clasificadores con el mejor
# rendimiento posible para los siguientes conjunto de datos:

# - Votos de congresistas US
# - Concesion de prestamos
# - Criticas de peliculas en IMDB (ver NOTA con instrucciones para obtenerlo)

# En todos los casos, sera necesario separar un conjunto de test para dar la
# valoracion final de los clasificadores obtenidos. Se permite usar
# train_test_split de Scikit Learn, para separar el conjunto de test y/o
# validacion. Ajustar tambien el valor del parametro de suavizado k. 

# Mostrar el proceso realizado en cada caso, y los rendimientos obtenidos. 
from sklearn.model_selection import train_test_split


def my_cross_validation(clf, X, y, cv, sklearn_clf=False):
    scores = []
    for i in range(cv):
        x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=random.randint(1, 10000), stratify=y)

        train_func = clf.fit if sklearn_clf else clf.entrena
        train_func(x_train, y_train)

        acc = rendimiento(clf, x_test, y_test, sklearn_clf=sklearn_clf)
        scores.append(acc)
    return scores


def select_best_clf(litsa):
    # TODO >>> Manue
    pass


# do experiments with the three datasets
# TODO >>> Manue  (si son demasiado largo, (que al ser varios conjuntos, puede ser), creo que quedaria mas claro en un archivo aparte)
# (como funciones que despues se puede llamar aqui)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# NOTA:
# INSTRUCCIONES PARA OBTENER EL CONJUNTO DE DATOS IMDB A USAR EN EL TRABAJO

# Este conjunto de datos se puede obtener y vectorizar usando lo visto en el
# Notebook del tema de modelos probabilisticos. No usar todo el conjunto: 
# extraer, usando random.sample, 2000 criticas en el conjunto de entrenamiento
# y 400 del conjunto de datos. Usar por ejemplo la siguiente secuencia de
# instrucciones, para extraer los textos:


# >>> import random as rd
# >>> from sklearn.datasets import load_files
# >>> reviews_train = load_files("data/aclImdb/train/")
# >>> muestra_entr=random.sample(list(zip(reviews_train.data,
#                                     reviews_train.target)),k=2000)
# >>> text_train=[d[0] for d in muestra_entr]
# >>> text_train = [doc.replace(b"<br />", b" ") for doc in text_train]
# >>> yimdb_train=np.array([d[1] for d in muestra_entr])
# >>> reviews_test = load_files("data/aclImdb/test/")
# >>> muestra_test=random.sample(list(zip(reviews_test.data,
#                                         reviews_test.target)),k=400)
# >>> text_test=[d[0] for d in muestra_test]
# >>> text_test = [doc.replace(b"<br />", b" ") for doc in text_test]
# >>> yimdb_test=np.array([d[1] for d in muestra_test])

# Ahora restaria vectorizar los textos. Puesto que la version NaiveBayes que
# se ha pedido implementar es la categorica (es decir, no es la multinomial),
# a la hora de vectorizar los textos lo haremos simplemente indicando en cada
# componente del vector si el correspondiente termino del vocabulario ocurre
# (1) o no ocurre (0). Para ello, usar CountVectorizer de Scikit Learn, con la
# opcion binary=True. Para reducir el numero de caracteristicas (es decir,
# reducir el vocabulario), usar "stop words" y min_df=50. Se puede ver como
# hacer esto en el Notebook del tema de modelos probabilisticos.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  















# =====================================================
# PARTE II: MODELOS LINEALES PARA CLASIFICACIoN BINARIA
# =====================================================

# En esta SEGUNDA parte se pide implementar en Python un clasificador binario
# lineal, basado en regresion logistica. 

# ---------------------------------------------
# II.1) Generando conjuntos de datos aleatorios
# ---------------------------------------------

# Previamente a la implementacion del clasificador lineal, conviene tener
# una funcion que genere aleatoriamente conjuntos de datos ficticios.  En
# concreto, se pide implementar una funcion:

# * Funcion genera_conjunto_de_datos_c_l_s(rango,dim,n_datos,prop=0.1): 

#   Debe devolver dos arrays numpy X e y, generados aleatoriamente. X debe
#   tener un "shape" (n_datos,dim) y sus valores deben ser numeros entre
#   -rango y rango. El array y debe tener la clasificacion binaria (1 o 0) de
#   cada ejemplo del conjunto X, en el mismo orden. El conjunto de datos debe
#   ser "casi" linealmente separable.

#   Para ello, generar en primer lugar un hiperplano aleatorio (mediante sus
#   coeficientes, elegidos aleatoriamente entre -rango y rango, e incluyendo
#   w0). Luego generar aleatoriamente cada ejemplo de igual manera y
#   clasificarlo como 1 o 0 dependiendo del lado del hiperplano en el que se
#   situe. Eso aseguraria que el conjunto de datos fuera linealmente
#   separable. Por ultimo, cambiar de clase a una proporcion pequena (dada por
#   el parametro prop) del total de ejemplos
from auxiliars.utils import plot_1Ddata


def get_random_sample(rango, *dims):
    return np.random.rand(*dims)*2*rango - rango


def genera_conjunto_de_datos_c_l_s(rango,dim,n_datos,prop=0.1):
    # hiperplane ecuation h = m_1*x_1 + ... + m_dim*x_dim.
    m = get_random_sample(rango, dim)

    x = get_random_sample(rango, n_datos, dim)
    hs = get_random_sample(rango, n_datos)

    hy = (x*m).sum(axis=1)
    y = hy >= hs
    if dim == 1:
        plot_1Ddata(m, x, hs, y, rango)

    # selecting random indexes (avoiding repetitions)
    random_indexes = np.random.permutation(np.arange(10))[:int(n_datos*prop)]
    y[random_indexes] = ~y[random_indexes]
    if dim == 1:
        plot_1Ddata(m, x, hs, y, rango)

    return x, y.astype(np.int)


# ---------------------------------------------
# II.2) Implementacion de un clasificador lineal
# ---------------------------------------------

# En esta seccion se pide implementar un clasificador BINARIO basado en
# regresion logistica, con algoritmo de entrenamiento de descenso por el
# gradiente mini-batch (para minimizar la entropia cruzada).

# En concreto se pide implementar una clase: 

# class RegresionLogisticaMiniBatch():

#     def __init__(self,clases=[0,1],normalizacion=False,
#                  rate=0.1,rate_decay=False,batch_tam=64)
#         .....
        
#     def entrena(self,X,y,n_epochs,reiniciar_pesos=False):

#         .....        

#     def clasifica_prob(self,ejemplo):

#         ......
    
#     def clasifica(self,ejemplo):
                        
#          ......

### TODO >> en el archivo auxiliars/logistic_regresion.py

# Explicamos a continuacion cada uno de estos elementos:


# * El constructor tiene los siguientes argumentos de entrada:

#   + Una lista clases (de longitud 2) con los nombres de las clases del
#     problema de clasificacion, tal y como aparecen en el conjunto de datos. 
#     Por ejemplo, en el caso de los datos de las votaciones, esta lista seria
#     ["republicano","democrata"]. La clase que aparezca en segundo lugar de
#     esta lista se toma como la clase positiva.  

#   + El parametro normalizacion, que puede ser True o False (False por
#     defecto). Indica si los datos se tienen que normalizar, tanto para el
#     entrenamiento como para la clasificacion de nuevas instancias. La
#     normalizacion es la estandar: a cada caracteristica se le resta la media
#     de los valores de esa caracteristica en el conjunto de entrenamiento, y
#     se divide por la desviacion tipica de dichos valores.

#  + rate: si rate_decay es False, rate es la tasa de aprendizaje fija usada
#    durante todo el aprendizaje. Si rate_decay es True, rate es la
#    tasa de aprendizaje inicial. Su valor por defecto es 0.1.

#  + rate_decay, indica si la tasa de aprendizaje debe disminuir en
#    cada epoch. En concreto, si rate_decay es True, la tasa de
#    aprendizaje que se usa en el n-esimo epoch se debe de calcular
#    con la siguiente formula: 
#       rate_n= (rate_0)*(1/(1+n)) 
#    donde n es el numero de epoch, y rate_0 es la cantidad introducida
#    en el parametro rate anterior. Su valor por defecto es False. 

#  + batch_tam: indica el tamano de los mini batches (por defecto 64) que se
#    usan para calcular cada actualizacion de pesos.



# * El metodo entrena tiene los siguientes parametros de entrada:

#  + X e y son los datos del conjunto de entrenamiento y su clasificacion
#    esperada, respectivamente. El primero es un array con los ejemplos, y el
#    segundo un array con las clasificaciones de esos ejemplos, en el mismo
#    orden.

#  + n_epochs: numero de veces que se itera sobre todo el conjunto de
#    entrenamiento.

#  + reiniciar_pesos: si es True, cada vez que se llama a entrena, se
#    reinicia al comienzo del entrenamiento el vector de pesos de
#    manera aleatoria (tipicamente, valores aleatorios entre -1 y 1).
#    Si es False, solo se inician los pesos la primera vez que se
#    llama a entrena. En posteriores veces, se parte del vector de
#    pesos calculado en el entrenamiento anterior, excepto que se diera
#    explicitamente el vector de pesos en el parametro peso_iniciales.  

#  + pesos_iniciales: si no es None y el parametro anterior reiniciar_pesos no
#    es False, es un array con los pesos iniciales. Este parametro puede ser
#    util para empezar con unos pesos que se habian obtenido y almacenado como
#    consecuencia de un entrenamiento anterior.



# * Los metodos clasifica y clasifica_prob se describen como en el caso del
#   clasificador NaiveBayes. Igualmente se debe devolver
#   ClasificadorNoEntrenado si llama a los metodos de clasificacion antes de
#   entrenar. 

# Se recomienda definir la funcion sigmoide usando la funcion expit de
# scipy.special, para evitar "warnings" por "overflow":

# from scipy.special import expit    
#
# def sigmoide(x):
#    return expit(x)



# Ejemplo de uso (usando la funcion del apartado anterior para generar el
# conjunto de datos). Probarlo con varios de estos conjuntos generados
# aleatoriamente:

from auxiliars.logistic_regresion import RegresionLogisticaMiniBatch
# -------------------------------------------------------------
# >>> X1,y1=genera_conjunto_de_datos_c_l_s(4,8,400)
X1,y1 = genera_conjunto_de_datos_c_l_s(4,8,400)

# El 25% para test:
# >>> X1e,y1e=X1[:300],y1[:300]
# >>> X1t,y1t=X1[300:],y1[300:]
X1e,y1e=X1[:300],y1[:300]
X1t,y1t=X1[300:],y1[300:]

# >>> clas_pb1=RegresionLogisticaMiniBatch([0,1],rate=0.1, rate_decay=true)
# >>> clas_pb1.entrena(X1e,y1e,10000)
clas_pb1 = RegresionLogisticaMiniBatch([0,1], rate=0.1, rate_decay=True, normalizacion=True)
print("\n ++ Training on random data for Logistic Regresion ++ \n")
clas_pb1.entrena(X1e, y1e, 100, bias=False)

# Clasificamos un ejemplo de test, y lo comparamos con su clase real:
# >>> clas_pb1.clasifica(X1t[0]),y1t[0]
# >>> (1, 1)

# Comprobamos el rendimiento sobre entrenamiento y prueba:
# >>> rendimiento(clas_pb1,X1e,y1e)
# 0.8733333333333333
# >>> rendimiento(clas_pb1,X1t,y1t)
# 0.83


# ----------------------------------------------------------------


# Otro ejemplo, con los datos del cancer de mama, que se puede cargar desde
# Scikit Learn:

# >>> from sklearn.datasets import load_breast_cancer
# >>> cancer=load_breast_cancer()
print("\n ++ Training on breast cancer (sklearn) for Logistic Regresion ++ \n")

from sklearn.datasets import load_breast_cancer
cancer=load_breast_cancer()
# >>> X_cancer,y_cancer=cancer.data,cancer.target
X_cancer,y_cancer = cancer.data,cancer.target
Xe_cancer, Xt_cancer, ye_cancer, yt_cancer = train_test_split(X_cancer, y_cancer, random_state=42, stratify=y_cancer)

# >>> lr_cancer=RegresionLogisticaMiniBatch(rate=0.1,rate_decay=True,normalizacion=True)
lr_cancer = RegresionLogisticaMiniBatch(rate=0.1,rate_decay=True,normalizacion=True)
# >>> lr_cancer.entrena(Xe_cancer,ye_cancer,10000)
lr_cancer.entrena(Xe_cancer, ye_cancer, 10000, bias=False)

# >>> rendimiento(lr_cancer,Xe_cancer,ye_cancer)
# 0.9906103286384976
# >>> rendimiento(lr_cancer,Xt_cancer,yt_cancer)
# 0.972027972027972

# -----------------------------------------------------------------







# -----------------------------------
# II.3) Aplicando Regresion Logistica 
# -----------------------------------

# Usando el clasificador implementado, obtener clasificadores con el mejor
# rendimiento posible para los siguientes conjunto de datos:

# - Votos de congresistas US
# - Cancer de mama 
# - Criticas de peliculas en IMDB

# Como antes, sera necesario separar conjuntos de validacion y test para dar
# la valoracion final de los clasificadores obtenidos Se permite usar
# train_test_split de Scikit Learn para esto. Ajustar los parametros de tamano
# de batch, tasa de aprendizaje y rate_decay. En alguno de los conjuntos de
# datos puede ser necesaria normalizacion.

# Mostrar el proceso realizado en cada caso, y los rendimientos obtenidos. 





















# ===================================
# PARTE III: CLASIFICACIoN MULTICLASE
# ===================================

# Se pide implementar un algoritmo de regresion logistica para problemas de
# clasificacion en los que hay mas de dos clases, usando  la tecnica
# de "One vs Rest" (OvR)

# ------------------------------------
# III.1) Implementacion de One vs Rest
# ------------------------------------


#  En concreto, se pide implementar una clase python RL_OvR con la siguiente
#  estructura, y que implemente un clasificador OvR usando como base el
#  clasificador binario del apartado anterior.

from auxiliars.one_vs_rest import RL_OvR

from sklearn.datasets import load_iris
# class RL_OvR():

#     def __init__(self,clases,rate=0.1,rate_decay=False,batch_tam=64):

#        ......

#     def entrena(self,X,y,n_epochs):

#        .......

#     def clasifica(self,ejemplo):

#        ......
            



#  Los parametros de los metodos significan lo mismo que en el apartado
#  anterior, excepto que ahora "clases" puede ser una lista con mas de dos
#  elementos. 

#  Un ejemplo de sesion, con el problema del iris:

iris=load_iris()
X_iris=iris.data
y_iris=iris.target
Xe_iris,Xt_iris,ye_iris,yt_iris = train_test_split(X_iris,y_iris)

print("\n ++ Training One vs Rest ++ \n")

rl_iris = RL_OvR([0,1,2], rate=0.001, batch_tam=20)
rl_iris.entrena(Xe_iris,ye_iris,n_epochs=100)

# --------------------------------------------------------------------
# >>> from sklearn.datasets import load_iris
# >>> iris=load_iris()
# >>> X_iris=iris.data
# >>> y_iris=iris.target
# >>> Xe_iris,Xt_iris,ye_iris,yt_iris=train_test_split(X_iris,y_iris)

# >>> rl_iris=RL_OvR([0,1,2],rate=0.001,batch_tam=20)

# >>> rl_iris.entrena(Xe_iris,ye_iris,n_epochs=1000)

# >>> rendimiento(rl_iris,Xe_iris,ye_iris)
# 0.9732142857142857

# >>> rendimiento(rl_iris,Xt_iris,yt_iris)
# >>> 0.9736842105263158
# --------------------------------------------------------------------


# ------------------------------------------------------------
# III.2) Clasificacion de imagenes de digitos escritos a mano
# ------------------------------------------------------------


#  Aplicar la implementacion del apartado anterior, para obtener un
#  clasificador que prediga el digito que se ha escrito a mano y que se
#  dispone en forma de imagen pixelada, a partir de los datos que estan en el
#  archivo digidata.zip que se suministra.  Cada imagen viene dada por 28x28
#  pixeles, y cada pixel vendra representado por un caracter "espacio en
#  blanco" (pixel blanco) o los caracteres "+" (borde del digito) o "#"
#  (interior del digito). En nuestro caso trataremos ambos como un pixel negro
#  (es decir, no distinguiremos entre el borde y el interior). En cada
#  conjunto las imagenes vienen todas seguidas en un fichero de texto, y las
#  clasificaciones de cada imagen (es decir, el numero que representan) vienen
#  en un fichero aparte, en el mismo orden. Sera necesario, por tanto, definir
#  funciones python que lean esos ficheros y obtengan los datos en el mismo
#  formato numpy en el que los necesita el clasificador. 

#  Los datos estan ya separados en entrenamiento, validacion y prueba. Si el
#  tiempo de computo en el entrenamiento no permite terminar en un tiempo
#  razonable, usar menos ejemplos de cada conjunto.

# Ajustar los parametros de tamano de batch, tasa de aprendizaje y
# rate_decay para tratar de obtener un rendimiento aceptable (por encima del
# 75% de aciertos sobre test). 

