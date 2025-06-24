import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler , OneHotEncoder , LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os
import sys
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from sklearn.impute import SimpleImputer
from src.utilitys import save_object
# from src.components.data_ingestion import data_ingestion_initiate


@dataclass

class DataTransformationConfig:
    preprocesser_obj_file_path = os.path.join("Artifacts" , "preprocesser.pkl")
 

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def data_transformation_obj(self):

        try:
            num_columns = [
                "person_age",
                "person_income",
                "person_emp_exp",
                "loan_amnt",
                "loan_int_rate",
                "loan_percent_income",
                "cb_person_cred_hist_length",
                "credit_score"
            ]

            

            cat_columns =["person_gender" , 
                        "person_education" , 
                        "person_home_ownership" , 
                        "loan_intent" , "previous_loan_defaults_on_file"]


            num_pipeline = Pipeline(
                steps=[
                    ("imputer" , SimpleImputer(strategy="mean")),
                    ("standard_scaling" , StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer" , SimpleImputer(strategy="most_frequent")),
                    ("encoding" , OneHotEncoder()),
                    ("standard_scaling" , StandardScaler(with_mean = False))
                ]
            )
            logging.info(f"deviding numarical columns{num_columns}")
            logging.info(f"deviding cat columns {cat_columns}")

            preprocesser = ColumnTransformer(
                [
                        ("num_pipeline" , num_pipeline , num_columns) ,
                        ("cat_pipeline" , cat_pipeline , cat_columns)
                ]

            )
            return preprocesser
        

        except Exception as e:
            raise CustomException(e , sys)
    
    def initate_data_transforming(self,train_data , test_data):


        try:
            train_data = pd.read_csv(train_data)
            test_data = pd.read_csv(test_data)

            logging.info("read train and test data successfully ")

            preprocessing_obj = self.data_transformation_obj()
            logging.info("preprocessing object is defined ")

            target_variable = "loan_status"

            input_feauter_train_data = train_data.drop(columns = ['loan_status'] , axis = 1)
            target_feauter_train_data = train_data["loan_status"]

            input_feauter_test_data = test_data.drop(columns = ["loan_status"] , axis = 1)
            target_feauter_test_data = test_data["loan_status"]

            logging.info("preprocessing on training data and testing data")

            input_feauter_train_arr = preprocessing_obj.fit_transform(input_feauter_train_data)
            input_feauter_test_arr = preprocessing_obj.fit_transform(input_feauter_test_data)

            train_arr = np.c_[
                input_feauter_train_arr , np.array(target_feauter_train_data)
            ]

            test_arr = np.c_[
                input_feauter_test_arr , np.array(target_feauter_test_data)
            ]
            logging.info("saved preprocesssing object ")

            save_object(

                file_path = self.data_transformation_config.preprocesser_obj_file_path,
                obj = preprocessing_obj

            )
            # train_path =os.makedirs(os.path.dirname(self.data_transformation_config.preprocesser_obj_file_path_train) , exist_ok=True)
            # preprocessed_train_data = train_arr.to_ssv(train_path)
            # test_path = os.makedirs(os.path.dirname(self.data_transformation_config.preprocesser_obj_file_path_test) , exist_ok= True)
            # preprocessed_test_data = test_arr.to_csv(test_path)


            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocesser_obj_file_path
                # preprocesser_obj_file_path.preprocesser_obj_file_path
            )


        except Exception as e:
            raise CustomException(e ,sys)
        


