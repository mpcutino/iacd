class ClasificadorNoEntrenado(Exception): pass


def not_trained_decorator(f):
    def work(self, ejemplo):
        if not self.is_trained:
            raise ClasificadorNoEntrenado
        else:
            return f(self, ejemplo)
    return work


class BaseClassifier():

    def entrena(self,X,y):
        pass

    @not_trained_decorator
    def clasifica_prob(self,ejemplo):
        pass

    @not_trained_decorator
    def clasifica(self,ejemplo):
        probs = self.clasifica_prob(ejemplo)

        rc, rp = -1, -100000
        for c, p in probs.items():
            if p > rp:
                rp = p
                rc = c
        return rc
