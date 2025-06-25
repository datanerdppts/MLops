from src.logger import logging
from src.exception import CustomException
from src.utilitys import load_object
import os
import sys
import pandas as pd

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self , feauters):

        try:

            model_file_path = os.path.join("Artifacts" , "model.pkl")
            preprocesser_file_path = os.path.join("Artifacts" , "preprocesser.pkl")

            model = load_object(file_path=model_file_path)
            preprocesser = load_object(file_path=preprocesser_file_path)

            scaled_feauters = preprocesser.transform(feauters)
            predctions = model.predict(scaled_feauters)

            return predctions
        except Exception as e:
            raise CustomException(e , sys)
        

class CustomData:

    def __init__(self,
                 person_age,	person_gender,	
                 person_education,	person_income,	
                 person_emp_exp,	person_home_ownership,	
                 loan_amnt,	loan_intent,	loan_int_rate,	
                 loan_percent_income,	cb_person_cred_hist_length,	
                 credit_score,	previous_loan_defaults_on_file
                 
                 
                 ):
        self.person_age = person_age
        self.person_gender = person_gender
        self.person_education = person_education
        self.person_income = person_income
        self.person_emp_exp = person_emp_exp
        self.person_home_ownership = person_home_ownership
        self.loan_amnt = loan_amnt
        self.loan_intent = loan_intent
        self.loan_int_rate = loan_int_rate
        self.loan_percent_income = loan_percent_income
        self.cb_person_cred_hist_length = cb_person_cred_hist_length
        self.credit_score = credit_score
        self.previous_loan_defaults_on_file = previous_loan_defaults_on_file


    def get_dataframe(self):
        try:
                
            data = {
                "person_age" : [self.person_age],
                "person_gender" : [self.person_gender],
                "person_education" : [self.person_education],
                "person_income" : [self.person_income],
                "person_emp_exp" : [self.person_emp_exp],
                "person_home_ownership" : [self.person_home_ownership],
                "loan_amnt" : [self.loan_amnt],
                "loan_intent" : [self.loan_intent],
                "loan_int_rate" : [self.loan_int_rate],
                "loan_percent_income" :[self.loan_percent_income],
                "cb_person_cred_hist_length" : [self.cb_person_cred_hist_length],
                "credit_score" : [self.credit_score],
                "previous_loan_defaults_on_file" : [self.previous_loan_defaults_on_file]    
            }

            return pd.DataFrame(data)
        except Exception as e:
            raise CustomException(e , sys)






