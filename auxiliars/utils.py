import numpy as np
import matplotlib.pyplot as plt


def plot_1Ddata(m, x, hs, positive_cond, rango):
    hx = np.arange(-rango, rango+0.5, step=0.5)
    hy = hx*m
    plt.plot(hx, hy)

    plt.scatter(x[positive_cond], hs[positive_cond], c="red", label="positive")
    plt.scatter(x[~positive_cond], hs[~positive_cond], c="green", label="negative")

    plt.legend()
    plt.show()
    plt.close()
