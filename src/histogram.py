import sys as sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ft_statistics import ft_variance


def load(path: str) -> pd.DataFrame:
    """Load a DataBase with a path (str) and return it
       Raise:
           TypeError()"""
    if not isinstance(path, str):
        raise TypeError("Bad argument type -> need path: str")
    dt = pd.read_csv(path)
    dt.dropna(inplace=True)
    return dt

def histogram(db: pd.DataFrame):
    """Load an historigram of the most homogeneous feature"""
    tmp = list(db.select_dtypes(include=['float64']).columns)
    homogeneous = tmp[0]
    variance = ft_variance(list(db[tmp[0]]));
    for i in range(len(tmp)):
        if ft_variance(list(db[tmp[i]])) <= variance:
            variance = ft_variance(list(db[tmp[i]]))
            homogeneous = tmp[i]
    plt.hist(db[homogeneous], bins=10)
    plt.title(homogeneous)
    plt.show()

def main():
    try:
        assert len(sys.argv) == 2, "Bad arguments need one"
        db_name = sys.argv[1]
        db = load(db_name)
        histogram(db)

    except AssertionError as e:
        print(f"AssertionError :{e}")

if __name__ == "__main__":
    main()
