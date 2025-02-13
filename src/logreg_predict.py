import pandas as pd
import sys as sys
import numpy as np
from pandas.core.interchange.dataframe_protocol import Column
from logisticRegression import LogisticRegression


def get_w_b_by_house(w_b, house):
    """Extracts weights and bias from a DataFrame for a specific house."""
    line = w_b.loc[w_b["House"] == house]
    weights_str = line["Weights"].values[0]
    weights = np.array(list(map(float, weights_str.split(","))))
    bias = line["Bias"].values[0]
    return weights, bias


def main():
    try:
        assert len(sys.argv) == 3, "Bad arguments need one"
        assert sys.argv[1] == "datasets/dataset_test.csv", "datasets/dataset_test.csv is required"
        assert sys.argv[2] == "weights_bias.csv", "weights_bias.csv is required"
        houses = ["Slytherin", "Ravenclaw", "Gryffindor", "Hufflepuff"]

        dbTest = pd.read_csv(sys.argv[1])
        Xtest = dbTest.drop(columns=["Hogwarts House"])
        Xtest = Xtest.select_dtypes(include=['float64'])
        Xtest = Xtest.fillna(0)
        w_b = pd.read_csv(sys.argv[2])
        result = np.full(400, "moldu", dtype=object)
        model = LogisticRegression()
        for house in houses:
            weights, bias = get_w_b_by_house(w_b, house)
            y_test = model.predict(Xtest, weights, bias)
            for i in range(len(y_test)):
                if y_test[i] == 1:
                    result[i] = house
        result = pd.DataFrame(result, columns=["Hogwarts House"])
        result.to_csv("Houses.csv", index=True, index_label="Index")

    except AssertionError as a:
        print(f"AssertionError: {a}")
    except FileNotFoundError as f:
        print(f"FileNotFoundError: {f}")

if __name__ == "__main__":
    main()
