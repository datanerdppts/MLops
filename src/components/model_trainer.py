import numpy as np
import pandas as pd
import os
from src.exception import CustomException
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from src.utilitys import model_evolute , save_object
from sklearn.metrics import accuracy_score
from src.logger import logging


class ModelTrainerconfig:
    model_path = os.path.join("Artifacts" , "model.pkl")

class ModelTrainier:
    def __init__(self):
        self.model_trainer_config = ModelTrainerconfig()

    def initate_model_trainer(self , train_arr , test_arr):

        try:
            logging.info("splitting  the data for model ")
            x_train , y_train , x_test , y_test = (
                train_arr[: , :-1],
                train_arr[: , -1],
                test_arr[: , :-1],
                test_arr[: , -1]

            )
            logging.info("model are defined")
            model = {
                "SVC" : SVC(),

                "Logistic_Regression" : LogisticRegression(),

                "KNeighbors_Classifier" : KNeighborsClassifier(),

                "DecisionTree_Classifier" : DecisionTreeClassifier(),

                "RandomForest_Classifier" : RandomForestClassifier(),

                "GradientBoosting_Classifier" : GradientBoostingClassifier(),

                "Gaussian_NB" : GaussianNB()

            }
            logging.info("parameters ae set for model  ")

            params = {
                "SVC" : {"kernel" : ['linear', 'poly', 'rbf', 'sigmoid'],
                            "C": [0.1, 1, 10]
                        
                        },


                "Logistic_Regression" : {"penalty" : ['l1', 'l2', 'elasticnet', 'none']},

                "KNeighbors_Classifier" : {"n_neighbors" : [2,4,6,8] ,
                                        "weights" : ["uniform" , "distance"],
                                        "metric" : ["minkowski" , "euclidean" , "manhattan"]
                                        
                                        },


                "DecisionTree_Classifier" : {"criterion" : ['gini' , 'entropy' , 'log_loss'],
                                            "max_depth" : [5,9,7,10,15]
                                            
                                            },

                "RandomForest_Classifier" : {"n_estimators" : [5,7,9,12,14]},

                "GradientBoosting_Classifier" : {"n_estimators" : [4,5,6,7,8,9]},

                "Gaussian_NB" : {"var_smoothing": [1e-9, 1e-8, 1e-7]}
            }

            logging.info("train the model ")
            report:dict = model_evolute(x_train , y_train ,x_test , y_test , model , params)

            best_model_score = max(sorted(report.values()))
            best_model_name = list(report.keys())[
                    list(report.values()).index(best_model_score)
                ]
            logging.info("getting most best model ")
            best_model = model[best_model_name]

            if best_model_score <= 0.6:
                raise CustomException("no best model is found :")
            logging.info("save the model")
            save_object(
                self.model_trainer_config.model_path,
                best_model
            )
            logging.info("get the predictions from  model")
            prediction = best_model.predict(x_test)
            accuracy = accuracy_score(y_test , prediction)
            
            return accuracy
        except Exception as e:
            raise CustomException(e , sys)

            






