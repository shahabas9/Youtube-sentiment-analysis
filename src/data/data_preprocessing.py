import pandas as pd
import numpy as np
import os
import logging
import yaml
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string


logger= logging.getLogger("data_preprocessing")
logger.setLevel(logging.DEBUG)

console_handler=logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler=logging.FileHandler("preprocessing_errors.log")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('[%(asctime)s : %(name)s : %(levelname)s : %(message)s]')

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# Define the preprocessing function
def preprocess_comment(comment):
    try:
        # Convert to lowercase
        comment = comment.lower()

        # Remove trailing and leading whitespaces
        comment = comment.strip()

        # Remove newline characters
        comment = re.sub(r'\n', ' ', comment)

        # Remove non-alphanumeric characters, except punctuation
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        # Remove stopwords but retain important ones for sentiment analysis
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        logger.error(f"Error happend during preprocessing the comment {e}")
        return comment

def normalize_text(df):
    try:
        df['clean_comment'] = df['clean_comment'].apply(preprocess_comment)
        logger.debug(f"text normalization completed")
        return df
    except Exception as e:
        logger.error(f"error happend during text normalization {e}")
        raise

def save_data(train_data :pd.DataFrame,test_data :pd.DataFrame,data_path :str)->None:
    try:
        preprocessed_data_path = os.path.join(data_path,"preprocessed")
        logger.debug(f"Creating directory {preprocessed_data_path}")
        os.makedirs(preprocessed_data_path,exist_ok=True)
        logger.debug(f"Created directory {preprocessed_data_path} or already Exiting folder{preprocessed_data_path}")
        train_data.to_csv(os.path.join(preprocessed_data_path,"train.csv"),index=False)
        test_data.to_csv(os.path.join(preprocessed_data_path,"test.csv"),index=False)
        logger.debug(f"preprocessed data are saved in {preprocessed_data_path}")
    except Exception as e:
        logger.error(f"unexpected error during saving the preprocessed data  {e}")
        raise

def main():
    try:
        logger.debug("starting data preprocessing ........")

        train_data=pd.read_csv("./data/raw/train.csv")
        test_data=pd.read_csv("./data/raw/test.csv")
        
        train_preprocessed_data = normalize_text(train_data)
        test_preprocessed_data = normalize_text(test_data)

        save_data(train_preprocessed_data,test_preprocessed_data,"./data")
    except  Exception as e:
        logger.error(f"Error while completing the data preprocessing steps {e}")
        raise

if __name__ =="__main__":
    main()