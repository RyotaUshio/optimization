#!/usr/bin/env python
#coding=utf-8

import numpy as np
import continuous as cont

# つぎの最適化問題を解きたい。
# minimize f(x) subject to x \in \R^n

## 目的関数は、それ自身と勾配ベクトル（・ヘッセ行列）とを合わせたobjFuncオブジェクトとして表現される。
## 以下は単純な1変数関数の例。

# 目的関数それ自体
def func(x):
    return x[0]*x[0] + 2.0*x[0] + 1.0

# 勾配ベクトル
def grad(x):
    return np.array([2.0*x[0] + 2.0])

# ヘッセ行列
def hesse(x):
    return np.array([[2.0]])


# これらを用いてobjFuncオブジェクトを生成する
f = cont.objFunc(func, grad, hesse)
# 最適化問題オブジェクトを生成
prob = cont.problem(f)

# 初期点を設定
x0 = np.array([0.0])
# 最急降下法ソルバのオブジェクトを生成
solver = cont.gradientDescent()
# パラメータの設定(デフォルトのままでも可)
solver.eps = 1e-06
solver.rho = 0.7
# 停留点を求める
# x_star = solver(prob, x0)
# print(x_star)
