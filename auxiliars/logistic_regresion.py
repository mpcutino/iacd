from auxiliars.base import BaseClassifier


class RegresionLogisticaMiniBatch(BaseClassifier):

    def __init__(self,clases=[0,1],normalizacion=False,
                rate=0.1,rate_decay=False,batch_tam=64):
        self.W = []
        pass

    def entrena(self,X,y,n_epochs,reiniciar_pesos=False):
        # TODO Miguel
        pass        

    def clasifica_prob(self,ejemplo):
        # TODO Manue
        pass
