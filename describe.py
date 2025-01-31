import sys as sys
import pandas as pd

def load(path: str) -> pd.DataFrame:
    """Load a DataBase with a path (str) and return it
       Raise:
           TypeError()"""

    if not isinstance(path, str):
        raise TypeError("Bad argument type -> need path: str")
    dt = pd.read_csv(path)
    return dt

def describe():
    """"""

def main():
    """"""
    try:
        assert len(sys.argv) == 2, "Bad arguments need one"
        db_name = sys.argv[1]
        db = load(db_name)
        tamara = db.iloc[0, 6:]
        print(tamara)
    except AssertionError as e:
        print(f"AssertionError :{e}")

if __name__ == "__main__":
    main()
