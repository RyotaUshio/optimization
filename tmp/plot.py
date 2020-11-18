#from continuous import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

def grad(x):
    return np.array([-400 * x[0] * (x[1] - x[0]**2) + 2 * (x[0] - 1),
                     200 * (x[1] - x[0]**2)])

def hesse(x):
    return np.array([[400*(x[0]**2+2*x[0]-x[1]) + 2,
                      -400*x[0]],
                      [-400*x[0],
                      200]])

def func(x, y):
    return 100*(y - x**2)**2 + (1 - x)**2

path = "~/2020A/最適化手法/ws/src/optimization/tmp"

for method, fname in zip(["Gradient Descent", "Newton's Method", "Quasi-Newton Method"],
                  ["gradient_descent", "newtons_method", "quasi_newton_method"]):
        
    filename = f"{path}/{fname}_log.out"
    df = pd.read_csv(filename, header=None, sep="\s+")
    
    x1, x2 = df[0].values, df[1].values
    range1 = x1.max() - x1.min()
    range2 = x2.max() - x2.min()
    xlim = (x1.min() - range1*0.2, x1.max() + range1*0.2)
    ylim = (x2.min() - range2*0.2, x2.max() + range2*0.2)

    n = 51
    x = np.linspace(xlim[0], xlim[1], n)
    y = np.linspace(ylim[0], ylim[1], n)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)

    plt.close("all")
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    
    ax.plot_surface(X, Y, Z, cmap=plt.cm.coolwarm)#, antialiased=False)
    ax.set(xlabel="x1", ylabel="x2")

    ax.plot(x1, x2, func(x1, x2), c="k", linewidth=2, alpha=1, marker="o", markersize=4)

    fig.savefig(f"{fname}.png")



# x0 = np.array([1.2, 1.2])
# x0 = np.array([-1.2, 1])
# def break_condition(f, grad, hesse, x):
#     grad_norm = np.linalg.norm(grad(x))
#     print(f"grad_norm = {grad_norm}")
#     return (grad_norm < 0.001)
# x, k = gradient_descent(f, grad, x0, break_condition)
# print(x[-1])
