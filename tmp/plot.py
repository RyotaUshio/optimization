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

methods = ["Gradient Descent", "Newton's Method", "Quasi-Newton Method"]
fnames = ["gradient_descent", "newtons_method", "quasi_newton_method"]

plt.close("all")
fig = plt.figure(figsize=(9, 5))
plt.subplots_adjust(wspace=0.5, hspace=0.2)
fig.suptitle("Initial guess = $(1.2, 1.2)$", fontsize=16)

a = 1 # 0
ax2D = fig.add_subplot(2, 3+a, 4)
ax3D = fig.add_subplot(2, 3+a, 8, projection="3d")
i = 1

xlim = (-1.5, 1.5)
ylim = (-4, 3)
n = 51
x = np.linspace(xlim[0], xlim[1], n)
y = np.linspace(ylim[0], ylim[1], n)
X, Y = np.meshgrid(x, y)
Z = func(X, Y)

markersize=.1

levels = 20
ax2D.contour(X, Y, Z, levels=levels)
ax2D.plot(x1, x2, marker="o", markersize=markersize)
ax2D.set(xlabel="$x_1$", ylabel="$x_2$", title="Comparison")

ax3D.plot_surface(X, Y, Z, cmap=plt.cm.coolwarm)
ax3D.set(xlabel="$x_1$", ylabel="$x_2$")
ax3D.set_zlabel("$f(x_1, x_2)$", rotation=0)



for method, fname in zip(methods, fnames):
    filename = f"{path}/{fname}_log.out"
    df = pd.read_csv(filename, header=None, sep="\s+")
    df.columns = ["x1", "x2", "alpha"]
    x1, x2 = df.x1.values, df.x2.values
    alpha = df.alpha.values

    range1 = x1.max() - x1.min()
    range2 = x2.max() - x2.min()
    # xlim = (x1.min() - range1*0.2, x1.max() + range1*0.2)
    # ylim = (x2.min() - range2*0.2, x2.max() + range2*0.2)

    ax2d = fig.add_subplot(2, 3+a, i)

    ax2d.contour(X, Y, Z, levels=levels)
    ax2d.plot(x1, x2, marker="o")
    ax2d.set(xlabel="$x_1$", ylabel="$x_2$", title=method)
    
    ax3d = fig.add_subplot(2, 3+a, i+3+a, projection="3d")

    ax3d.plot_surface(X, Y, Z, cmap=plt.cm.coolwarm)#, antialiased=False)
    ax3d.set(xlabel="$x_1$", ylabel="$x_2$")
    ax3d.set_zlabel("$f(x_1, x_2)$", rotation=0)

    ax3d.plot(x1, x2, func(x1, x2), c="k", linewidth=2, alpha=1, marker="o", markersize=markersize)

    ax2D.plot(x1, x2, marker="o")
    ax3D.plot(x1, x2, func(x1, x2), linewidth=2, alpha=1, marker="s", markersize=markersize)

    i += 1


