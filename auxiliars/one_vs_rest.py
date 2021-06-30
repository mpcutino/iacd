import numpy as np

from auxiliars.logistic_regresion import RegresionLogisticaMiniBatch


class RL_OvR():

    def __init__(self, clases, normalizacion=False, rate=0.1, rate_decay=False, batch_tam=64):
        self.classifiers = [
            RegresionLogisticaMiniBatch(clases, normalizacion=normalizacion, rate=rate, rate_decay=rate_decay, batch_tam=batch_tam) for _ in clases
        ]

    def entrena(self,X,y,n_epochs):
        for i, clf in enumerate(self.classifiers):
            print("\t == Training class {0} ==".format(i))
            # binarize data for class i
            yi = np.array(y)
            cond = yi == i
            yi[cond] = 1
            yi[~cond] = 0
            # train
            clf.entrena(X, yi, n_epochs, bias=False)

    def clasifica(self,ejemplo):
        pass
