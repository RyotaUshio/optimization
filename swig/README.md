SWIGによる外部関数インターフェース(FFI)
=================

C++で書かれた本ライブラリをPythonから呼び出すには、SWIGを利用する。

ビルド方法
-----
- Ubuntu 18.04
```
$ swig -c++ -python -I/usr/include/eigen3 continuous.i 
$ g++ -shared -fPIC -I./ -I/usr/include/python2.7 -I/usr/include/eigen3 -o _continuous.so continuous_wrap.cxx continuous.cpp -lm -lstdc++
```
- MacOS Mojave


インポート
---
```
$ python
Python 2.7.17 (default, Sep 30 2020, 13:38:04) 
[GCC 7.5.0] on linux2
Type "help", "copyright", "credits" or "license" for more information.
>>> import continuous
```
