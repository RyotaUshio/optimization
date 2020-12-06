# rates of convergence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
sns.set(style="ticks")
pd.options.display.precision = 14


read_csv = dict(header=None,
                names=["x1", "x2", "alpha"])

apnd = "_armijo"

grad = pd.read_csv(f"grad{apnd}.main", **read_csv)
newton = pd.read_csv(f"newton{apnd}.main", **read_csv)
quasi = pd.read_csv(f"quasi{apnd}.main", **read_csv)

x_star = np.array([1.0, 1.0])

for df in [grad, newton, quasi]:
    df["diff"] = (df[["x1", "x2"]] - x_star).apply(lambda row: np.linalg.norm(row), axis=1)  
    df["diffsq"] = df["diff"]**2

    diff = df["diff"]
    df["ratio"] = diff / pd.Series([np.nan]).append(diff, ignore_index=True)

    diffsq = df["diffsq"]
    df["ratiosq"] = diff / pd.Series([np.nan]).append(diffsq, ignore_index=True)
    
