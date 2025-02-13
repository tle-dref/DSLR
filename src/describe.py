import sys as sys
import pandas as pd
import numpy as np
import statistics as st
from ft_statistics import ft_mean, ft_min_max, ft_median, ft_quartile, ft_ecart_type
from histogram import load

class Feature:
    def __init__(self, lst: list, name:str):
        """Init the feature class to stock the statistics\n
        -name -count -mean -quartile 1 & 3 -std -min -max median"""
        self.name          = name
        self.count         = len(lst)
        self.mean          = ft_mean(lst)
        self.quartile      = ft_quartile(lst)
        self.std           = ft_ecart_type(lst)
        self.min, self.max = ft_min_max(lst)
        self.median        = ft_median(lst)


def describe(db: pd.DataFrame):
    """Display all feature in the Database in :.6f format"""
    tmp = list(db.select_dtypes(include=['float64']).columns)
    features = []
    for i in range(len(tmp)):
        features.append(Feature(list(db[tmp[i]]), tmp[i]))
    data = {}
    for feature in features:
        trunc_name = feature.name
        if len(feature.name) > 12:
            trunc_name = feature.name[:12] + '.'
        data[trunc_name] = [
            float(feature.count),
            float(feature.mean),
            float(feature.std),
            float(feature.min),
            float(feature.quartile[0]),
            float(feature.median),
            float(feature.quartile[1]),
            float(feature.max)
        ]
    df_describe = pd.DataFrame(data, index=["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"])
    print(df_describe.to_string(float_format="%.6f"))


def main():
    try:
        assert len(sys.argv) == 2, "Bad arguments need one"
        db_name = sys.argv[1]
        db = load(db_name)
        describe(db)
    except AssertionError as e:
        print(f"AssertionError :{e}")


if __name__ == "__main__":
    main()
