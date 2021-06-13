import numpy as np

from auxiliars.base import BaseClassifier
from auxiliars.utils import cross_entropy_gradient, cross_entropy_loss, sigmoid


class RegresionLogisticaMiniBatch(BaseClassifier):

    def __init__(self, clases=[0,1], normalizacion=False,
                rate=0.1, rate_decay=False, batch_tam=64):
        self.classes = np.arange(len(clases))
        self.class_names = clases
        self.batch_size = batch_tam
        self.rate = rate
        
        self.W = None


    def entrena(self, X, y, n_epochs, reiniciar_pesos=False, pesos_iniciales=None, bias=True):
        if reiniciar_pesos or self.W is None:
            if reiniciar_pesos and pesos_iniciales is not None:
                self.W = pesos_iniciales
            else:
                self.W, X = self.__init_W__(bias, X)

        # TODO Miguel
        for n in range(n_epochs):
            running_loss = 0
            batches = 0
            for i in range(0, len(X), self.batch_size):
                x_b, y_b = X[i:i+self.batch_size], y[i:i+self.batch_size]

                input_ = np.matmul(x_b, self.W)
                preds = sigmoid(input_)

                loss = cross_entropy_loss(preds, y_b)
                grad = cross_entropy_gradient(preds, y_b, x_b)

                self.W += self.rate*grad
                batches += 1
                # average loss per minibatch
                running_loss += loss.sum()/len(x_b)
            print("Average loss: {0:.4f}".format(running_loss/batches))


    def clasifica_prob(self,ejemplo):
        # TODO Manue
        pass

    @staticmethod
    def __init_W__(bias, X):
        if bias:
            # add 1s
            new_x = np.ones((X.shape[0], X.shape[1] + 1))
            new_x[:, 1:] = X
            X = new_x
        W = np.random.rand(X.shape[1])
        return W, X
