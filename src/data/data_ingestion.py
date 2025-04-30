import pandas as pd
import numpy as np
import os
import logging
import yaml
from sklearn.model_selection import train_test_split

logger= logging.getLogger("data_ingestion")
logger.setLevel(logging.DEBUG)

console_handler=logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler=logging.FileHandler("error.log")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('[%(asctime)s : %(name)s : %(levelname)s : %(message)s]')

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path : str) -> dict:
    try:
        with open(params_path,'r') as file:
            params = yaml.safe_load(file)
        logger.debug(f"parameter received from {params_path}")
        return params
    except FileNotFoundError:
        logger.error(f"file not found {params_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"yaml error {e}")
        raise
    except Exception as e:
        logger.error(f"unexpected error {e}")
        raise

def load_data(data_url : str)->pd.DataFrame:
    try:
        data = pd.read_csv(data_url)
        logger.debug(f"data loded from the url {data_url}")
        return data
    except pd.errors.ParserError as e:
        logger.error(f"Failed to parse the csv {e}")
        raise
    except Exception as e:
        logger.error(f"unexpected error {e}")
        raise

def preprocess_data(df :pd.DataFrame)->pd.DataFrame:
    try:
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        df=df[(df['clean_comment'].str.strip() != '')]
        logger.debug(f"Data preprocessing completed: handles missing values,duplicates and empty strings")
        return df
    except KeyError as e:
        logger.error(f"Missing column in the dataframe : {e}")
        raise
    except Exception as e:
        logger.error(f"unexpected error {e}")

def save_data(train_data :pd.DataFrame,test_data :pd.DataFrame,data_path :str)->None:
    try:
        raw_data_path = os.path.join(data_path,"raw")
        os.makedirs(raw_data_path,exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path,"train.csv"),index=False)
        test_data.to_csv(os.path.join(raw_data_path,"test.csv"),index=False)
        logger.debug(f"train data and test data are saved in {raw_data_path}")
    except Exception as e:
        logger.error(f"unexpected error during saving the data  {e}")
        raise

def main():
    try:
        params = load_params("../../params.yaml")
        test_size=params["data_ingestion"]["test_size"]
        url="https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv"
        df = load_data(url)
        final_df = preprocess_data(df)
        train_data,test_data = train_test_split(final_df,test_size=test_size,random_state=42)
        save_data(train_data,test_data,"../../data/")
    except Exception as e:
        logger.error(f"failed to complete data ingestion process {e}")
        raise

if __name__ =="__main__":
    main()



    