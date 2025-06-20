import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
import os
import sys



@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join("Artifacts" , "train_data.csv")
    test_data_path = os.path.join("Artifacts" , "test_data.csv")
    raw_data_path = os.path.join("Artifacts" , "data.csv")

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def data_ingestion_initiate(self):

        try:
            logging.info("Data ingestion initiation start")
            data = pd.read_csv("notebook/Data/loan_data.csv")
            logging.info("read data from data set ")
            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path) , exist_ok=True)
            data.to_csv(self.data_ingestion_config.raw_data_path , header=True , index=False)
            logging.info("split data data into trainig and testing")
            train_data , test_data = train_test_split(data , test_size=0.2 , random_state=42)
            train_data.to_csv(self.data_ingestion_config.train_data_path , index = False , header = True)
            test_data.to_csv(self.data_ingestion_config.test_data_path , index = False , header = True)
            logging.info("data is splitted and store in Artifacts folder ")
        except Exception as e:
            raise CustomException(e , sys)
        

if __name__ == "__main__":
    obj = DataIngestion()
    obj.data_ingestion_initiate()