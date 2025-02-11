from histogram import load
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

def display_subject_histograms(db: pd.DataFrame):
    """
    For each subject (numeric column), display a histogram of the grade distribution
    by "Hogwarts House". The goal is to visualize which subjects show distribution
    differences among the houses, which may help in selecting relevant variables for
    logistic regression.

    Parameters:
      - db : DataFrame containing the data, with a "Hogwarts House" column and numeric columns for the grades.
    """
    subjects = db.select_dtypes(include=['float64']).columns
    n_subjects = len(subjects)

    ncols = 3
    nrows = math.ceil(n_subjects / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 4))

    if n_subjects == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for ax, subject in zip(axes, subjects):
        sns.histplot(
            data=db,
            x=subject,
            hue="Hogwarts House",
            multiple="dodge",
            palette="Set2",
            edgecolor="black",
            alpha=0.7,
            ax=ax
        )
        ax.set_title(f"Grade Distribution in {subject}")
        ax.set_xlabel("Grade")
        ax.set_ylabel("Frequency")

    for ax in axes[len(subjects):]:
        ax.remove()

    plt.tight_layout()
    plt.show()

def main():
    try:
        db = load("datasets/dataset_train.csv")
        display_subject_histograms(db)
    except Exception as e:
        print(f"Error : {e}")

if __name__ == "__main__":
    main()
