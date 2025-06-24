from src.exception import CustomException
from src.logger import logging
import sys

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score , accuracy_score

import pickle
import dill
import os
from tqdm import tqdm

def save_object(file_path , obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)


        
    except Exception as e:
        raise CustomException(e , sys)
    



# def model_evolute(x_train , y_train , x_test , y_test , model , params):
#     try:

#         score = {}

#         for i in range(len(list(model))):
#             model = list(model.values())[i]
#             params = params[list(model.keys())[i]]

#             gc = GridSearchCV(model, params, cv=4)
#             gc.fit(x_train , y_train)

#             model.set_params(**gc.best_params_)
#             model.fit(x_train , y_train)


#             y_train_predictions = model.predict(x_train)
#             y_test_prediction = model.predict(x_test)

#             y_train_score = accuracy_score(y_train , y_train_predictions)
#             y_test_score = accuracy_score(y_test , y_test_prediction)

#             score[list(model.keys())[i]] = y_test_score

#         return score

#     except Exception as e:
#         raise CustomException(e , sys)



def model_evolute(x_train, y_train, x_test, y_test, models, params):
    try:
        score = {}

        for model_name in tqdm(models):
            model_obj = models[model_name]
            param_grid = params[model_name]

            gc = GridSearchCV(model_obj, param_grid, cv=4)
            gc.fit(x_train, y_train)

            model_obj.set_params(**gc.best_params_)
            model_obj.fit(x_train, y_train)

            y_train_pred = model_obj.predict(x_train)
            y_test_pred = model_obj.predict(x_test)

            y_train_score = accuracy_score(y_train, y_train_pred)
            y_test_score = accuracy_score(y_test, y_test_pred)

            score[model_name] = y_test_score

        return score

    except Exception as e:
        raise CustomException(e, sys)
