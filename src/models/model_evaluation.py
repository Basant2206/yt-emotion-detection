import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import json

# fetch model
def load_model(model_path: str) -> np.binary_repr:
    try:
        with open(model_path, 'rb') as file:
            clf = pickle.load(file)
        return clf
    except FileNotFoundError:
        print(f"Error: The model file {model_path} was not found.")
        raise
    except pickle.UnpicklingError:
        print(f"Error: There was an issue unpickling the model {model_path}.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while loading the model: {e}")
        raise

# Data fetching from processed
def fetch_data(test_path: str) -> pd.DataFrame:
    try:
        test_data = pd.read_csv(test_path)
        return test_data
    except FileNotFoundError:
        print(f"Error: The test file {test_path} was not found.")
        raise
    except pd.errors.EmptyDataError:
        print(f"Error: The file {test_path} is empty.")
        raise
    except pd.errors.ParserError:
        print(f"Error: There was an issue parsing the file {test_path}.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while fetching the data: {e}")
        raise

# Model prediction function
def model_prediction(clf: np.binary_repr, X_test: np.array) -> np.array:
    try:
        y_pred = clf.predict(X_test)
        # y_pred_proba = clf.predict_proba(X_test)[:, 1]  # Uncomment if needed
        return y_pred
    except Exception as e:
        print(f"Error: An issue occurred during model prediction: {e}")
        raise

# Calculate evaluation metrics
def evaluate_metrics(y_test: np.array, y_pred: np.array) -> float:
    try:
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)  # Ensure `y_pred` are the predicted labels
        accuracy = accuracy_score(y_test, y_pred)
        return precision, recall, auc, accuracy
    except ValueError as e:
        print(f"Error: Invalid value encountered during metric evaluation: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred during evaluation: {e}")
        raise

# Save metrics to a JSON file
def save_metrics(precision: float, recall: float, auc: float, accuracy: float) -> None:
    try:
        metric_dict = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'auc': auc}
        with open('metric.json', 'w') as f:
            json.dump(metric_dict, f, indent=4)
    except PermissionError:
        print("Error: Permission denied to write to 'metric.json'.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while saving metrics: {e}")
        raise

def main():
    try:
        clf = load_model('model.pkl')
        test_data = fetch_data("./data/features/test_bow.csv")
        
        if test_data.empty:
            print("Error: The test data is empty. Please check the input file.")
            raise ValueError("Empty test data.")
        
        X_test = test_data.iloc[:, 0:-1].values
        y_test = test_data.iloc[:, -1].values
        
        y_pred = model_prediction(clf, X_test)
        precision, recall, auc, accuracy = evaluate_metrics(y_test, y_pred)
        
        save_metrics(precision, recall, auc, accuracy)
        
    except Exception as e:
        print(f"An error occurred during execution: {e}")

if __name__ == "__main__":
    main()
