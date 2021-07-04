from collections import Counter

import math
import numpy as np

from auxiliars.base import BaseClassifier


class NaiveBayes(BaseClassifier):

    def __init__(self, k=1):
        super(NaiveBayes, self).__init__()
        self.k = float(k)

        self.a_priori_probs = {}
        self.conditional_probs = {}
         
    def entrena(self,X,y):
        assert len(X.shape) == 2

        count = Counter(y)
        self.a_priori_probs = {c: math.log(count[c]/len(y)) for c in count}

        for i in range(X.shape[1]):
            atribute_ = X[:, i]
            atribute_and_class = zip(atribute_, y)
            atribute_count = Counter(atribute_and_class)

            kA = self.k*len(np.unique(atribute_))
            # this are the probabilities for known values
            known_probs = {(a, c): math.log((n + self.k)/(count[c] + kA)) for (a, c), n in atribute_count.items()}
            # may be there are some unkonw probabilities, because they do not exist on the data
            full_probs = self.__fill_probs(known_probs, y, kA)
            self.conditional_probs[i] = full_probs
        # print(self.conditional_probs)

        self.is_trained = True

    def clasifica_prob(self,ejemplo):
        res = {}
        for c in self.a_priori_probs:
            log_conditional_prob = 0
            for i, value in enumerate(ejemplo):
                atribute_probs = self.conditional_probs[i]
                log_conditional_prob += atribute_probs[(value, c)] if (value, c) in atribute_probs else 0
            res[c] = self.a_priori_probs[c] + log_conditional_prob
        res = {c: math.e**res[c] for c in res}
        total_ = sum(res.values())
        return {c: res[c]/total_ for c in res}
        # return res

    def __fill_probs(self, known_probs, y, kA):
        full_probs = dict(known_probs)
        classes_ = np.unique(y)
        count = Counter(y)
        atributes = np.unique([a for (a, _) in known_probs])
        for a in atributes:
            for c in classes_:
                if (a, c) in known_probs: continue
                full_probs[(a, c)] = math.log(self.k / (count[c] + kA))
        return full_probs
