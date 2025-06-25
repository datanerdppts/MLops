# from flask import Flask ,render_template , request

# import pandas as pd
# import numpy as np

# from sklearn.preprocessing import StandardScaler
# from src.pipeline.prediction_pipline import CustomData , PredictPipeline



# app = Flask(__name__)

# @app.route('/predictdata' , methods = ['POST' , 'GET'])

# def predict_data_point():
    
#     if request.method == 'GET':
#         return render_template('home.html')
#     else:
#         custom_data = CustomData(
#                 person_age = request.form.get("person_age"),
#                 person_gender = request.form.get("person_gender"),
#                 person_education = request.form.get("person_education"),
#                 person_income = request.form.get("person_income"),
#                 person_emp_exp = request.form.get("person_emp_exp"),
#                 person_home_ownership = request.form.get("person_home_ownership"),
#                 loan_amnt = request.form.get("loan_amnt"),
#                 loan_intent = request.form.get("loan_intent"),
#                 loan_int_rate = request.form.get("loan_int_rate"),
#                 loan_percent_income = request.form.get("loan_percent_income"),
#                 cb_person_cred_hist_length = request.form.get("cb_person_cred_hist_length"),
#                 credit_score = request.form.get("credit_score"),
#                 previous_loan_defaults_on_file = request.form.get("previous_loan_defaults_on_file"),
#         )

#         pred_df = custom_data.get_dataframe()
#         print(pred_df)
#         print("Before Prediction")

#         predict_pipeline=PredictPipeline()
#         print("Mid Prediction")
#         results=predict_pipeline.predict(pred_df)
#         print("after Prediction")
#         return render_template('predict_page.html',results=results[0])


# if __name__ == "__main__":
#     app.run(debug=True,host='0.0.0.0')




from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from src.pipeline.prediction_pipline import CustomData, PredictPipeline

app = Flask(__name__)

@app.route('/predictdata', methods=['POST', 'GET'])
def predict_data_point():
    if request.method == 'GET':
        return render_template('predict_page.html', results=None)
    else:
        custom_data = CustomData(
            person_age=request.form.get("person_age"),
            person_gender=request.form.get("person_gender"),
            person_education=request.form.get("person_education"),
            person_income=request.form.get("person_income"),
            person_emp_exp=request.form.get("person_emp_exp"),
            person_home_ownership=request.form.get("person_home_ownership"),
            loan_amnt=request.form.get("loan_amnt"),
            loan_intent=request.form.get("loan_intent"),
            loan_int_rate=request.form.get("loan_int_rate"),
            loan_percent_income=request.form.get("loan_percent_income"),
            cb_person_cred_hist_length=request.form.get("cb_person_cred_hist_length"),
            credit_score=request.form.get("credit_score"),
            previous_loan_defaults_on_file=request.form.get("previous_loan_defaults_on_file"),
        )

        pred_df = custom_data.get_dataframe()
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        print("Prediction result:", results)

        return render_template('predict_page.html', results=results[0])

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
