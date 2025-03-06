import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import yaml
import logging

# logging configure
logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

#file handler
file_handler = logging.FileHandler('errors.log')
file_handler.setLevel('ERROR')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> float:
    try:
        with open(params_path, 'r') as file:
            test_size = yaml.safe_load(file)['data_ingestion']['test_size']
        logger.debug('test size set level')
        return test_size
    except Exception as e:
        logger.error('file Not found')
        return 0.2  # default value

def read_data(url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        print(f"Error reading data: {e}")
        return pd.DataFrame()  # return empty DataFrame on error

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df.drop(columns=['tweet_id'], inplace=True)
        df = df.dropna()
        final_df = df[df['sentiment'].isin(['happiness', 'sadness'])]
        final_df['sentiment'].replace({'happiness': 1, 'sadness': 0}, inplace=True)
        return final_df
    except Exception as e:
        print(f"Error processing data: {e}")
        return pd.DataFrame()  # return empty DataFrame on error

def save_data(data_path: str, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    try:
        os.makedirs(data_path, exist_ok=True)
        train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)
    except Exception as e:
        print(f"Error saving data: {e}")

def main():
    try:
        test_size = load_params('params.yaml')
        df = read_data('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
        if df.empty:
            print("Dataframe is empty. Exiting...")
            return

        final_df = process_data(df)
        if final_df.empty:
            print("Processed dataframe is empty. Exiting...")
            return

        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
        data_path = os.path.join("data", "raw")
        save_data(data_path, train_data, test_data)
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    main()
