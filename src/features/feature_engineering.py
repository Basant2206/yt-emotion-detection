import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
import yaml

def load_params(params_path: str) -> int:
    try:
        with open(params_path, 'r') as file:
            max_features = yaml.safe_load(file)['feature_engineering']['max_features']
        return max_features
    except FileNotFoundError:
        print(f"Error: The file {params_path} was not found.")
        raise
    except KeyError as e:
        print(f"Error: Missing expected key {e} in {params_path}.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while loading params: {e}")
        raise

# data fetche from processed
def fetch_data(train_path: str, test_path: str) -> pd.DataFrame:
    try:
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        return train_data, test_data
    except FileNotFoundError as e:
        print(f"Error: {e}")
        raise
    except pd.errors.EmptyDataError:
        print(f"Error: One of the files is empty. Please check {train_path} or {test_path}.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while fetching data: {e}")
        raise

def feature_engg(X_train: np.array, X_test: np.array, max_features: int) -> np.array:
    try:
        # Apply Bag of Words (CountVectorizer)
        vectorizer = CountVectorizer(max_features=max_features)

        # Fit the vectorizer on the training data and transform it
        X_train_bow = vectorizer.fit_transform(X_train)
        # Transform the test data using the same vectorizer
        X_test_bow = vectorizer.transform(X_test)
        return X_train_bow, X_test_bow
    except ValueError as e:
        print(f"Error: Invalid value encountered in feature engineering: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred during feature engineering: {e}")
        raise

def save_data(data_path: str, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    try:
        os.makedirs(data_path, exist_ok=True)  # Ensure the directory exists
        train_df.to_csv(os.path.join(data_path, "train_bow.csv"), index=False)
        test_df.to_csv(os.path.join(data_path, "test_bow.csv"), index=False)
    except PermissionError:
        print(f"Error: Permission denied to write to {data_path}.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while saving data: {e}")
        raise

def main():
    try:
        max_features = load_params('params.yaml')
        train_data, test_data = fetch_data("./data/interim/train_processed.csv", "./data/interim/test_processed.csv")

        train_data.fillna(" ", inplace=True)
        test_data.fillna(" ", inplace=True)
        
        # separate label
        X_train = train_data['content'].values
        y_train = train_data['sentiment'].values
        X_test = test_data['content'].values
        y_test = test_data['sentiment'].values

        X_train_bow, X_test_bow = feature_engg(X_train, X_test, max_features)

        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df['label'] = y_train
        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df['label'] = y_test

        # Store features
        data_path = os.path.join("data", "features")
        save_data(data_path, train_df, test_df)
    except Exception as e:
        print(f"An error occurred during execution: {e}")

if __name__ == "__main__":
    main()
