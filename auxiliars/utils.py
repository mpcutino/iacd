import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit


def plot_1Ddata(m, x, hs, positive_cond, rango):
    hx = np.arange(-rango, rango+0.5, step=0.5)
    hy = hx*m
    plt.plot(hx, hy)

    plt.scatter(x[positive_cond], hs[positive_cond], c="red", label="positive")
    plt.scatter(x[~positive_cond], hs[~positive_cond], c="green", label="negative")

    plt.legend()
    plt.show()
    plt.close()


def mse_loss_function(preds, truth):
    return ((truth - preds)**2).sum()


def mse_loss_gradient(preds, truth, x):
    # -2 should mutiply the equation, but we remove it because it is not necessary
    # then, we mus remember that we must sum the gradient (normally you subtract to move on the oposite direction, 
    # but we remove the negative value, because minus by minus is positive)
    return (x*(truth -  preds)).sum(axis=1)


def sigmoid(z):
    return expit(z)


def cross_entropy_loss(pred, truth, correction_value=0.0001):
    # prevent logarithmic omputation of 0
    pred[pred == 1] -= correction_value
    pred[pred == 0] += correction_value
    return -truth*np.log(pred) - (1 - truth)*np.log(1 - pred)


def cross_entropy_gradient(pred, truth, X):
    # the gradient is with a minus, but we dont use to sum after instead of subtract
    return np.dot(truth-pred, X)

# BEGIN >>> for digit data
def converter(l):
    aux = [0 if i == " " else 1 for i in l]
    return aux

def load_images(data_path):
    data = np.loadtxt(data_path, dtype=str, delimiter='\n', comments='\n')
    numeric_images = [converter(i) for i in data]
    # print(numeric_images[0])
    res = np.array(numeric_images, dtype=np.uint8)
    ammount_data = len(numeric_images)/28
    return res.reshape(int(ammount_data), 28*28)

def load_labels(data_path):
    data = np.loadtxt(data_path, dtype=np.int32, delimiter='\n')
    return data
# END >>> for digit data
