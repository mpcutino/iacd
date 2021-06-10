
import numpy
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder

from auxiliars.naive_bayes import NaiveBayes
from clasificadores import rendimiento


# my custom Naive Bayes    
weather = ['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny','Sunny', 'Rainy','Sunny','Overcast','Overcast','Rainy']
temp = ['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild']

play = ['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']


X_data = numpy.array([weather, temp]).transpose()

nb_clima = NaiveBayes(k=1)
nb_clima.entrena(X_data, play)

ejemplo = ['Overcast', 'Mild']
# ejemplo = ['Rainy', 'Mild']

print(nb_clima.clasifica(ejemplo))

# sklearn Naive Bayes
wle = LabelEncoder()
tle = LabelEncoder()
y_encod = LabelEncoder()

w, t = wle.fit_transform(weather),  tle.fit_transform(temp)
y = y_encod.fit_transform(play)

clf = CategoricalNB()
sklearn_X = numpy.array([w, t]).transpose()
clf = clf.fit(sklearn_X, y)

print(clf.predict(numpy.array([[0, 2]])))

acc = rendimiento(nb_clima, X_data, play)
print("My custom Naive Bayes accuracy >>> {0}".format(acc))

acc2 = rendimiento(clf, sklearn_X, y, sklearn_clf=True)
print("Sklearn Naive Bayes accuracy >>> {0}".format(acc2))
