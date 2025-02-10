import sys as sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from describe import load, Feature

def find_similar(features: list):
    """find the lowest minimum between the mean of all features"""
    f1, f2 = "", ""
    diff_mean = 0
    for i in range(len(features)):
        tmp_f = features[i]
        j = 0
        for j in range(len(features)):
            if (tmp_f.name == features[j].name):
                continue
            if abs(tmp_f.mean - features[j].mean) < diff_mean or diff_mean == 0:
                diff_mean = abs(tmp_f.mean - features[j].mean)
                f1 = tmp_f
                f2 = features[j]
    return f1, f2


def scatter_plot(db: pd.DataFrame):
    """"""
    tmp = list(db.select_dtypes(include=['float64']).columns)
    features = []
    for i in range(len(tmp)):
        features.append(Feature(list(db[tmp[i]]), tmp[i]))
    f1, f2 = find_similar(features)
    mat1 = db[f1.name]
    mat2 = db[f2.name]
    plt.plot(mat1, marker='o', ls='', color='orange')
    plt.plot(mat2, marker='o', ls='', color='red')
    plt.xlabel(f1.name)
    plt.ylabel(f2.name)
    plt.title(f"{f1.name} / {f2.name}")
    plt.show()

def main():
    """"""
    try:
        assert len(sys.argv) == 2, f"Bad Argument need one -> python3 {sys.argv[0]} 'pathToDb'"
        db = load(sys.argv[1])
        scatter_plot(db)
    # except TypeError as t:
    #     print(f"TypeError: {t}")
    except AssertionError as a:
        print(f"AssertionError: {a}")

if __name__ == "__main__":
    main()
