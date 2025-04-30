import pandas as pd
import numpy as np
import os
import logging
import yaml
import pickle
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer


logger= logging.getLogger("model_building")
logger.setLevel(logging.DEBUG)

console_handler=logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler=logging.FileHandler("model_building_errors.log")
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

def load_data(data_path : str)->pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
        df.fillna("",inplace=True)
        logger.debug(f"data loded from the url {data_path}")
        return df
    except pd.errors.ParserError as e:
        logger.error(f"Failed to parse the csv {e}")
        raise
    except Exception as e:
        logger.error(f"unexpected error {e}")
        raise

def apply_tfidf(train_data : pd.DataFrame,max_features : int,ngram_range : tuple)->tuple:
    try:
        vectorizer = TfidfVectorizer(max_features=max_features,ngram_range=ngram_range)
        
        X_train = train_data['clean_comment'].values
        y_train = train_data["category"].values

        X_train_tfidf = vectorizer.fit_transform(X_train)
        logger.debug(f"TF-IDF transformation complete, train shape : {X_train_tfidf.shape}")
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.abspath(os.path.join(script_dir, "../../"))
        with open(os.path.join(root_dir,'tfidf_vectorizer.pkl'),'wb') as f:
            pickle.dump(vectorizer,f)
        
        logger.debug(f"TF-IDF applied with trigramd and data transformed")
        return X_train_tfidf,y_train
    except Exception as e:
        logger.error(f"error during tf-idf transformation {e}")
        raise

def train_lgb(X_train:np.ndarray,y_train : np.ndarray,learning_rate:float,max_depth :int,n_estimators:int)->lgb.LGBMClassifier:
    try:
        best_model = lgb.LGBMClassifier(
            objective ='multiclass',
            num_class = 3,
            metric = 'multi_logloss',
            is_unbalance = True,
            class_weight = 'balanced',
            reg_alpha=0.1,
            reg_lambda=0.1,
            learning_rate=learning_rate,
            max_depth = max_depth,
            n_estimators = n_estimators
        )
        best_model.fit(X_train,y_train)
        logger.debug("training LightGBM model completed")
        return best_model
    except Exception as e:
        logger.error(f"Error occured during training {e}")
        raise

def save_model(model,model_path:str)->str:
    try:
        with open(model_path,"wb") as file:
            pickle.dump(model,file)
        logger.debug(f"MOdel saved to {model_path}")
    except Exception as e:
        logger.error(f"Error during while saving the data,{e}")
        raise

def main():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        params_path = os.path.abspath(os.path.join(script_dir, "../../params.yaml"))
        params = load_params(params_path)
        
        max_features=params['model_building']['max_feature']
        ngram_range = tuple(params['model_building']['ngram_range'])
        learning_rate = params['model_building']['learning_rate']
        max_depth = params['model_building']['max_depth']
        n_estimator = params['model_building']['n_estimator']


        preprocessed_data_path =os.path.abspath(os.path.join(script_dir, "../../data/preprocessed/train.csv")) 
        train_data = load_data(preprocessed_data_path)
        X_train_tfidf,y_train =apply_tfidf(train_data,max_features,ngram_range)
        best_model = train_lgb(X_train_tfidf,y_train,learning_rate,max_depth,n_estimator)
        root_dir = os.path.abspath(os.path.join(script_dir, "../../"))
        save_model(best_model,os.path.join(root_dir,"ligtgbm_model.pkl"))
    except Exception as e:
        logger.error(f"error occured during feature engineering and model build process ,{e}")
        raise

if __name__ =="__main__":
    main()