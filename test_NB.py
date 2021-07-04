
import numpy as np
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder

from auxiliars.naive_bayes import NaiveBayes


def rendimiento(clasificador, X, y, sklearn_clf=False):
    preds = list(map(clasificador.clasifica, X)) if not sklearn_clf else clasificador.predict(X)

    preds = np.array(preds)
    y = np.array(y)

    acc = (preds == y).sum()
    
    return acc/len(y)

# my custom Naive Bayes    
# weather = ['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny','Sunny', 'Rainy','Sunny','Overcast','Overcast','Rainy']
# temp = ['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild']

# play = ['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']


# X_data = numpy.array([weather, temp]).transpose()

# nb_clima = NaiveBayes(k=1)
# nb_clima.entrena(X_data, play)

# ejemplo = ['Overcast', 'Mild']
# # ejemplo = ['Rainy', 'Mild']

# print(nb_clima.clasifica(ejemplo))

# # sklearn Naive Bayes
# wle = LabelEncoder()
# tle = LabelEncoder()
# y_encod = LabelEncoder()

# w, t = wle.fit_transform(weather),  tle.fit_transform(temp)
# y = y_encod.fit_transform(play)

# clf = CategoricalNB()
# sklearn_X = numpy.array([w, t]).transpose()
# clf = clf.fit(sklearn_X, y)

# print(clf.predict(numpy.array([[0, 2]])))

# acc = rendimiento(nb_clima, X_data, play)
# print("My custom Naive Bayes accuracy >>> {0}".format(acc))

# acc2 = rendimiento(clf, sklearn_X, y, sklearn_clf=True)
# print("Sklearn Naive Bayes accuracy >>> {0}".format(acc2))


from auxiliars.utils import load_images, load_labels
from auxiliars.one_vs_rest import RL_OvR


X_train = load_images("data/digitdata/trainingimages")
y_train = load_labels("data/digitdata/traininglabels")

X_valid = load_images("data/digitdata/validationimages")
y_valid = load_labels("data/digitdata/validationlabels")

X_test = load_images("data/digitdata/testimages")
y_test = load_labels("data/digitdata/testlabels")


# print("Train size")
# print(X_train.shape)
# print(y_train.shape)

# print("Validation size")
# print(X_valid.shape)
# print(y_valid.shape)

# print("Test size")
# print(X_test.shape)
# print(y_test.shape)

# X_train = np.stack(X_train, X_valid)
# y_train = np.stack(y_train, y_valid)

def rl_ovr_grid_search(clases, params, train_tuple, validation_tuple, n_epochs=300):
    print("Grid search for {0} epochs...".format(n_epochs))
    best_params, best_score = {}, -1
    for p in params:
        clf = RL_OvR(clases, **p)
        clf.entrena(train_tuple[0], train_tuple[1], n_epochs=n_epochs)
        score = rendimiento(clf, validation_tuple[0], validation_tuple[1])
        if score > best_score:
            best_score = score
            best_params = p
    print("Best score {0} for params {1}".format(best_score, best_params))
    return best_params


params_ = [
    {'rate': 0.001, 'batch_tam': 64, 'rate_decay': False},
    {'rate': 0.001, 'batch_tam': 64, 'rate_decay': True},
    {'rate': 0.01, 'batch_tam': 32, 'rate_decay': True},
    {'rate': 0.001, 'batch_tam': 128, 'rate_decay': False},
    {'rate': 0.1, 'batch_tam': 128, 'rate_decay': True},
]
clases_ = np.arange(1, 10, 1)

best_p = rl_ovr_grid_search(clases_, params_, (X_train, y_train), (X_valid, y_valid), n_epochs=75)

rl_digit = RL_OvR(clases_, **best_p)
xe, ye = np.vstack([X_train, X_valid]), np.hstack([y_train, y_valid])

print("\nTraining on full data (using both training and validation)")
rl_digit.entrena(xe, ye, n_epochs=1000)

print("Rendimiento en dataset digit (training, test)")
res = rendimiento(rl_digit, xe, ye)
print(res)
res = rendimiento(rl_digit, X_test, y_test)
print(res)
