import sys as sys
import pandas as pd
import numpy as np
import statistics as st
from ft_statistics import ft_mean, ft_min_max, ft_median, ft_quartile, ft_ecart_type

class Feature:
    def __init__(self, lst: list, name:str):
        """"""
        self.name          = name
        self.count         = len(lst)
        self.mean          = ft_mean(lst)
        self.quartile      = ft_quartile(lst)
        self.std           = ft_ecart_type(lst)
        self.min, self.max = ft_min_max(lst)
        self.median        = ft_median(lst)
    def print_var(self):
        print(f"Feature    = {self.name}")
        print(f"Count      = {self.count}")
        print(f"Mean       = {self.mean}")
        print(f"Std        = {self.std}")
        print(f"Min        = {self.min}")
        print(f"25%        = {self.quartile[0]}")
        print(f"Median     = {self.median}")
        print(f"75%        = {self.quartile[1]}")
        print(f"Max        = {self.max}")

def load(path: str) -> pd.DataFrame:
    """Load a DataBase with a path (str) and return it
       Raise:
           TypeError()"""

    if not isinstance(path, str):
        raise TypeError("Bad argument type -> need path: str")
    dt = pd.read_csv(path)
    return dt

def describe(db: pd.DataFrame):
    """"""
    tmp = list(db.select_dtypes(include=['float64']).columns)
    features = []
    for i in range(len(tmp)):
        features.append(Feature(list(db[tmp[i]]), tmp[i]))
    features[0].print_var()

def main():
    """"""
    try:
        assert len(sys.argv) == 2, "Bad arguments need one"
        db_name = sys.argv[1]
        db = load(db_name)
        describe(db)
    except AssertionError as e:
        print(f"AssertionError :{e}")

if __name__ == "__main__":
    main()
