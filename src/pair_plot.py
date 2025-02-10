from histogram import load
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def pair_plot(db: pd.DataFrame):
    """Display a pair plot of the most homogeneous feature"""
    tmp = list(db.select_dtypes(include=['float64']).columns)
    data = db.loc[:, tmp].copy()

    data.columns = [col[:6] for col in data.columns]
    sns.pairplot(db, hue="Hogwarts House", diag_kind="hist", corner=True)
    plt.show()

def main():
    try:
        db = load("datasets/dataset_train.csv")
        pair_plot(db)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
