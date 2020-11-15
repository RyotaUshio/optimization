SWIGによる外部関数インターフェース(FFI)
=================

C++で書かれた本ライブラリをPythonから呼び出すには、SWIGを利用する。

ビルド方法
-----
```
$ make
```
とすればよい。デフォルトではPython2用のモジュールが出力され、Python3では利用できない。Python3から呼び出したい場合には、Makefileに定義されている変数PYTHON_VERSIONを明示的に指定する。
```
$ make PYTHON_VERSION=3.6
```
こうすればPython3からインポートできる。もちろん、
```
$ make PYTHON_VERSION=２.７
```
とすることもできる(自分の環境では2.7か3.6のみ対応)。

インポート
---
```
$ python
Python 2.7.17 (default, Sep 30 2020, 13:38:04) 
[GCC 7.5.0] on linux2
Type "help", "copyright", "credits" or "license" for more information.
>>> import continuous
```
