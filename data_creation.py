from pathlib import Path

from sklearn.datasets import load_iris


OUTPUT_PATH = Path("data/iris.csv")


def main():
    iris = load_iris(as_frame=True)
    dataframe = iris.frame

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(OUTPUT_PATH, index=False)

    print(f"Dataset written to: {OUTPUT_PATH}")
    print(f"Data rows: {len(dataframe)}")
    print(f"Columns: {len(dataframe.columns)}")


if __name__ == "__main__":
    main()