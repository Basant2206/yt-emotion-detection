import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingClassifier
import yaml

def load_params(params_path: str) -> np.__dict__:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)['model_building']
        return params
    except FileNotFoundError:
        print(f"Error: The parameter file {params_path} was not found.")
        raise
    except KeyError as e:
        print(f"Error: Missing expected key {e} in {params_path}.")
        raise
    except yaml.YAMLError as e:
        print(f"Error: There was an issue parsing the YAML file: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while loading params: {e}")
        raise

# Data fetching function
def fetch_data(data_path: str) -> pd.DataFrame:
    try:
        train_data = pd.read_csv(data_path)
        return train_data
    except FileNotFoundError:
        print(f"Error: The file {data_path} was not found.")
        raise
    except pd.errors.EmptyDataError:
        print(f"Error: The file {data_path} is empty.")
        raise
    except pd.errors.ParserError:
        print(f"Error: There was an issue parsing the file {data_path}.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while fetching data: {e}")
        raise

# Define and train the GBoost model
def train_model(X_train: np.array, y_train: np.array, params: np.__dict__) -> np.binary_repr:
    try:
        clf = GradientBoostingClassifier(n_estimators=params['n_estimators'], learning_rate=params['learning_rate'])
        clf.fit(X_train, y_train)
        return clf
    except ValueError as e:
        print(f"Error: Invalid values encountered during model training: {e}")
        raise
    except KeyError as e:
        print(f"Error: Missing key in model parameters: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred during model training: {e}")
        raise

def save_model(clf: np.binary_repr) -> None:
    try:
        with open('model.pkl', 'wb') as file:
            pickle.dump(clf, file)
    except PermissionError:
        print("Error: Permission denied to write the model file.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while saving the model: {e}")
        raise

def main():
    try:
        params = load_params('params.yaml')
        train_data = fetch_data("./data/features/train_bow.csv")
        
        if train_data.empty:
            print("Error: The training data is empty. Please check the input file.")
            raise ValueError("Empty training data.")
        
        X_train = train_data.iloc[:, 0:-1].values
        y_train = train_data.iloc[:, -1].values
        
        clf = train_model(X_train, y_train, params)
        save_model(clf)
    except Exception as e:
        print(f"An error occurred during execution: {e}")

if __name__ == "__main__":
    main()
