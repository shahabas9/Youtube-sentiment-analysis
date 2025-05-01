import os
import json
import mlflow
import logging

logger= logging.getLogger("model_registartion")
logger.setLevel(logging.DEBUG)

console_handler=logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler=logging.FileHandler("model_registration_errors.log")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('[%(asctime)s : %(name)s : %(levelname)s : %(message)s]')

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

mlflow.set_tracking_uri("http://ec2-54-226-116-239.compute-1.amazonaws.com:5000/")

def load_model_info(file_path:str)->str:
    try:
        with open(file_path,"r") as file:
            model_info = json.load(file)
        logger.debug(f"loaded the model info from model path,{file_path}")
        return model_info
    except Exception as e:
        logger.error(f"error during while loading the model info ,{e}")
        raise

def register_model(model_name : str, model_info:dict):
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        model_version = mlflow.register_model(model_uri,model_name)
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name = model_name,
            version = model_version.version,
            stage = 'Staging'
        )
    except Exception as e:
        logger.error(f"error occured while model registry process,{e}")
        raise

def main():
    try:
        model_info_path = 'experiment_info.json'
        model_info = load_model_info(model_info_path)
        model_name = "yt_chrome_plugin_model"
        register_model(model_name,model_info)
    except Exception as e:
        logger.error(f" Error occured during registering the model {e}")
        raise

if __name__ =="__main__":
    main()