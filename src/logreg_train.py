import sys as sys
from histogram import load
import pandas as pd
import matplotlib.pyplot as plt

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
        assert len(sys.argv) == 2, "Bad arguments need one"
        db_name = sys.argv[1]
        house = "Slytherin"
        db = load(db_name)
        db = binary_house_name(db, house)
        print(db)
    except AssertionError as e:
        print(f"AssertionError :{e}")

if __name__ == "__main__":
    main()
