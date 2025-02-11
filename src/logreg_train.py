import sys as sys
from histogram import load
import pandas as pd
import matplotlib.pyplot as plt
from logisticRegression import LogisticRegression
import numpy as np

def binary_house_name(db: pd.DataFrame, house: str) -> pd.DataFrame:
    """change all house name to 1 if it's the house in parameters else 0"""
    if not isinstance(house, str):
        raise TypeError("Bad argument type -> need house: str")
    if not isinstance(db, pd.DataFrame):
        raise TypeError("Bad argument type -> need db: pd.DataFrame")
    if house not in db["Hogwarts House"].unique():
        raise ValueError("House not in the database")
    db["Hogwarts House"] = db["Hogwarts House"].apply(lambda x: 1 if x == house else 0)
    return db

def find_best_lr(db: pd.DataFrame):
    """Find the best learning rate for the logistic regression"""
    possible_lr = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    best_lr = 0
    best_score = 0
    db = binary_house_name(db, "Ravenclaw")
    y = db["Hogwarts House"]
    X = db.select_dtypes(include=['float64'])
    for lr in possible_lr:
        model = LogisticRegression(l_rate=lr)
        model.fit(X, y)
        score = model.score(X, y)
        if score > best_score:
            best_score = score
            best_lr = lr
    print(f"final best score : {best_score} with lr : {best_lr}")
    return best_lr

def main():
    try:
        assert len(sys.argv) == 3, "Bad arguments need one"
        db_name = sys.argv[1]
        db_test = sys.argv[2]
        houses = ["Slytherin", "Ravenclaw", "Gryffindor", "Hufflepuff"]
        result = np.array(["moldu"] * 400)
        db = load(db_name)
        lr = find_best_lr(db);
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
            model = LogisticRegression(l_rate=lr)
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
