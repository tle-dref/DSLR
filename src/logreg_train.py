import sys as sys
from histogram import load
import pandas as pd
import matplotlib.pyplot as plt
from logisticRegression import LogisticRegression
import numpy as np

def binary_house_name(db: pd.DataFrame, house: str) -> pd.DataFrame:
    """"""
    if not isinstance(house, str):
        raise TypeError("Bad argument type -> need house: str")
    if not isinstance(db, pd.DataFrame):
        raise TypeError("Bad argument type -> need db: pd.DataFrame")
    if house not in db["Hogwarts House"].unique():
        raise ValueError("House not in the database")
    db["Hogwarts House"] = db["Hogwarts House"].apply(lambda x: 1 if x == house else 0)
    return db


def main():
    try:
        assert len(sys.argv) == 3, "Bad arguments need one"
        db_name = sys.argv[1]
        db_test = sys.argv[2]
        houses = ["Slytherin", "Ravenclaw", "Gryffindor", "Hufflepuff"]
        result = np.array(["moldu"] * 400)

        for house in houses:
            db = load(db_name)
            dbTest = pd.read_csv(db_test)
            Xtest = dbTest.drop(columns=["Hogwarts House"])
            Xtest = Xtest.select_dtypes(include=['float64'])
            Xtest = Xtest.fillna(0)
            db = binary_house_name(db, house)
            y = db["Hogwarts House"]
            X = db.select_dtypes(include=['float64'])
            # print(X.shape)
            # print(dbTest)
            model = LogisticRegression()
            model.fit(X, y)
            y_test = model.predict(Xtest)
            # print(f"result : {y_test}")
            for i in range(len(y_test)):
                if y_test[i] == 1:
                    result[i] = house
                # if i == 375:
                    # print(y_test[i])
        print(f"result : {result}")
        # print(np.sum(y_test == y) / len(y))

    except AssertionError as e:
        print(f"AssertionError :{e}")

if __name__ == "__main__":
    main()
