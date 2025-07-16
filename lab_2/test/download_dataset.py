import pandas as pd


def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df[['text', 'label']]  # Keep only relevant columns
    df.columns = ['text', 'emotion']  # Rename columns for clarity
    return df


if __name__ == "__main__":
    file_path = "D:\\Users\\Legion\\datasets\\emotions.csv"
    df = load_data(file_path)
    print(df.head())
