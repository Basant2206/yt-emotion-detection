import numpy as np
import pandas as pd
import os
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

# fetch the data from raw
def fetch_data(train_path: str, test_path: str) -> pd.DataFrame:
    try:
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
    except FileNotFoundError as e:
        print(f"Error: File not found. Please check the paths. {e}")
        raise
    except pd.errors.EmptyDataError as e:
        print(f"Error: The file is empty. {e}")
        raise
    except Exception as e:
        print(f"An error occurred while reading files: {e}")
        raise
    return train_data, test_data


def lemmatization(text: str) -> str:
    try:
        lemmatizer = WordNetLemmatizer()
        text = text.split()
        text = [lemmatizer.lemmatize(y) for y in text]
        return " ".join(text)
    except Exception as e:
        print(f"Error during lemmatization: {e}")
        raise


def remove_stop_words(text: str) -> str:
    try:
        stop_words = set(stopwords.words("english"))
        text = [i for i in str(text).split() if i not in stop_words]
        return " ".join(text)
    except Exception as e:
        print(f"Error during stop words removal: {e}")
        raise


def removing_numbers(text: str) -> str:
    try:
        text = ''.join([i for i in text if not i.isdigit()])
        return text
    except Exception as e:
        print(f"Error during number removal: {e}")
        raise


def lower_case(text: str) -> str:
    try:
        text = text.split()
        text = [y.lower() for y in text]
        return " ".join(text)
    except Exception as e:
        print(f"Error during case conversion: {e}")
        raise


def removing_punctuations(text: str) -> str:
    try:
        text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
        text = text.replace('؛', "", )
        text = re.sub('\s+', ' ', text)
        text = " ".join(text.split())
        return text.strip()
    except Exception as e:
        print(f"Error during punctuation removal: {e}")
        raise


def removing_urls(text: str) -> str:
    try:
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)
    except Exception as e:
        print(f"Error during URL removal: {e}")
        raise


def remove_small_sentences(df):
    try:
        for i in range(len(df)):
            if len(df.text.iloc[i].split()) < 3:
                df.text.iloc[i] = np.nan
    except Exception as e:
        print(f"Error during small sentence removal: {e}")
        raise


def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df.content = df.content.apply(lambda content: lower_case(content))
        df.content = df.content.apply(lambda content: remove_stop_words(content))
        df.content = df.content.apply(lambda content: removing_numbers(content))
        df.content = df.content.apply(lambda content: removing_punctuations(content))
        df.content = df.content.apply(lambda content: removing_urls(content))
        df.content = df.content.apply(lambda content: lemmatization(content))
        #df.fillna(" ", inplace=True)
        return df
    except Exception as e:
        print(f"Error during text normalization: {e}")
        raise


def save_processed_data(data_path: str, train_processed_data: pd.DataFrame, test_processed_data: pd.DataFrame) -> None:
    try:
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"))
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"))
    except Exception as e:
        print(f"Error during saving processed data: {e}")
        raise


def main():
    try:
        nltk.download('wordnet')
        nltk.download('stopwords')
        train_data, test_data = fetch_data("./data/raw/train.csv", "./data/raw/test.csv")
        train_processed_data = normalize_text(train_data)
        test_processed_data = normalize_text(test_data)

        # Store data
        data_path = os.path.join("data", "processed")
        save_processed_data(data_path, train_processed_data, test_processed_data)

    except Exception as e:
        print(f"An error occurred in the main function: {e}")


if __name__ == "__main__":
    main()
