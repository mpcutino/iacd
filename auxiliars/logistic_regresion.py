import numpy as np

from auxiliars.base import BaseClassifier
from auxiliars.utils import cross_entropy_gradient, cross_entropy_loss, sigmoid


class RegresionLogisticaMiniBatch(BaseClassifier):

    def __init__(self, clases=[0,1], normalizacion=False,
                rate=0.1, rate_decay=False, batch_tam=64):
        super(RegresionLogisticaMiniBatch, self).__init__()
        self.classes = np.arange(len(clases))
        self.class_names = clases
        self.batch_size = batch_tam
        self.rate = rate

        self.W = None
        self.normalize_data = normalizacion
        self.rate_decay = rate_decay

        # normalization parameters
        self.std_norm = None
        self.mean_norm = None
        # bias training
        self.bias = False

        # training function (override this in child class to change the behaviour of the classifier)
        self.f_prediction = sigmoid
        self.f_loss = cross_entropy_loss
        self.f_loss_gradient = cross_entropy_gradient

    def entrena(self, X, y, n_epochs, reiniciar_pesos=False, pesos_iniciales=None, bias=True):
        if reiniciar_pesos or self.W is None:
            if reiniciar_pesos and pesos_iniciales is not None:
                self.W = pesos_iniciales
            else:
                self.W, X = self.__init_W__(bias, X)
                self.bias = bias
        if self.normalize_data:
            # normalizar los datos de entrada X
            X = np.array(X)
            self.std_norm = np.std(X, axis=0)
            self.std_norm[self.std_norm == 0] = 0.001   # to prevent a possible 0 division
            self.mean_norm = np.mean(X, axis=0)
            X = (X - self.mean_norm)/self.std_norm

        for n in range(n_epochs):
            running_loss = 0
            batches = 0
            rate_n = self.rate/(1+n) if self.rate_decay else self.rate
            for i in range(0, len(X), self.batch_size):

                x_b, y_b = X[i:i+self.batch_size], y[i:i+self.batch_size]

                input_ = np.matmul(x_b, self.W)
                preds = self.f_prediction(input_)

                loss = self.f_loss(preds, y_b)
                grad = self.f_loss_gradient(preds, y_b, x_b)

                self.W += rate_n*grad
                batches += 1
                # average loss per minibatch
                running_loss += loss.sum()/len(x_b)
                # print("Average loss: {0:.4f}".format(running_loss/batches))
        self.is_trained = True

    def clasifica_prob(self,ejemplo):
        if self.bias:
            ejemplo = np.append(1, ejemplo)
        
        if self.normalize_data:
            ejemplo = (np.array(ejemplo) - self.mean_norm)/self.std_norm

        input_ = np.dot(ejemplo, self.W)
        preds = self.f_prediction(input_)

        # print(preds)
        return {0:1-preds, 1:preds}

    @staticmethod
    def __init_W__(bias, X):
        if bias:
            # add 1s
            new_x = np.ones((X.shape[0], X.shape[1] + 1))
            new_x[:, 1:] = X
            X = new_x
        # inicializando con rango -1, 1
        W = 2*np.random.rand(X.shape[1]) - 1
        return W, X
