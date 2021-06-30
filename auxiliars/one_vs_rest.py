from auxiliars.logistic_regresion import RegresionLogisticaMiniBatch


class RL_OvR(RegresionLogisticaMiniBatch):

    def __init__(self,clases,rate=0.1,rate_decay=False,batch_tam=64):
        super(RegresionLogisticaMiniBatch, self).__init__()

        # change training functions: classification, loss and loss gradient

    def entrena(self,X,y,n_epochs):
        pass

    def clasifica(self,ejemplo):
        pass
