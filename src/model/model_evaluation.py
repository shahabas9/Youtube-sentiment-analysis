import pandas as pd
import numpy as np
import os
import logging
import yaml
import pickle
import mlflow
import mlflow.sklearn
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
from mlflow.models import infer_signature
from sklearn.feature_extraction.text import TfidfVectorizer


logger= logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)

console_handler=logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler=logging.FileHandler("model_evaluation_errors.log")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('[%(asctime)s : %(name)s : %(levelname)s : %(message)s]')

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

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

def load_model(model_path : str):
    try:
        with open(model_path,"rb") as file:
            model= pickle.load(file)
        logger.debug(f"model loaded from {model_path} ")
        return model
    except Exception as e:
        logger.error(f"unexpected error while loading the model, {e}")
        raise

def load_vectorizer(vectorizer_path : str):
    try:
        with open(vectorizer_path,"rb") as file:
            model= pickle.load(file)
        logger.debug(f"model loaded from {vectorizer_path} ")
        return model
    except Exception as e:
        logger.error(f"unexpected error while loading the vectorizer, {e}")
        raise

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

def evaluate_model(model,X_test:np.ndarray,y_test:np.ndarray):
    try:
        y_pred = model.predict(X_test)
        report = classification_report(y_test,y_pred,output_dict=True)
        cm = confusion_matrix(y_test,y_pred)
        logger.debug("model evalauation complete")
        return report,cm
    except Exception as e:
        logger.error(f"Error during model evaluation, {e}")
        raise

def log_confusion_matrix(cm,dataset_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix for  {dataset_name}")
    cm_file_path = f"confusion_matrix_{dataset_name}.png"
    plt.savefig(cm_file_path)
    mlflow.log_artifact(cm_file_path)
    plt.close()

def save_model_info(run_id:str,model_path:str,file_path:str)->None:
    try:
        model_info={
            'run_id' : run_id,
            'model_path' : model_path
        }
        with open(file_path,"w") as file:
            json.dump(model_info,file,indent=4)
        logger.debug(f"model info saved into the file path:{file_path}")
    except Exception as e:
        logger.error(f"error occured while saving the model info , {e}")
        raise


def main():
    mlflow.set_tracking_uri("http://ec2-54-226-116-239.compute-1.amazonaws.com:5000/")
    mlflow.set_experiment("dvc-pipeline-runs")
    with mlflow.start_run() as run:
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            root_dir = os.path.abspath(os.path.join(script_dir, "../../"))
            params =load_params(os.path.join(root_dir,'params.yaml'))

            for key, value in params.items():
                mlflow.log_param(key,value)
            
            model =load_model(os.path.join(root_dir,"ligtgbm_model.pkl"))
            vectorizer = load_vectorizer(os.path.join(root_dir,"tfidf_vectorizer.pkl"))

            test_data = load_data(os.path.join(root_dir,"./data/preprocessed/test.csv"))
            X_test_tfidf =vectorizer.transform(test_data['clean_comment'].values)
            y_test = test_data["category"].values

            input_example = pd.DataFrame(X_test_tfidf.toarray()[:5],columns=vectorizer.get_feature_names_out())
            signature = infer_signature(input_example,model.predict(X_test_tfidf[:5]))

            mlflow.sklearn.log_model(
                model,
                "lightgbm_model",
                signature= signature,
                input_example = input_example
            )
            model_path = "lightgbm_model"
            save_model_info(run.info.run_id,model_path,"experiment_info.json")

            mlflow.log_artifact(os.path.join(root_dir,"tfidf_vectorizer.pkl"))

            report,cm = evaluate_model(model,X_test_tfidf,y_test)

            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    mlflow.log_metric(f"test_{label}_precision", metrics['precision'])
                    mlflow.log_metric(f"test_{label}_recall", metrics['recall'])
                    mlflow.log_metric(f"test_{label}_f1-score", metrics['f1-score'])

            log_confusion_matrix(cm,"Test Data")
            mlflow.set_tag("model_type", "LightGBM")
            mlflow.set_tag("task", "Sentiment Analysis")
            mlflow.set_tag("dataset", "YouTube Comments")
        except Exception as e:
            logger.error(f"failed to complete model evaluation process, {e}")
            raise

if __name__ =="__main__":
    main()