# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 20:56:11 2021

@author: QiwenWang
"""
import re
import pandas as pd
from flask import Flask, render_template, request
import pickle

from pandas.core.dtypes.common import classes

app = Flask(__name__, template_folder='templates', static_folder="st")


def convert_number(string):
    b = re.sub(r"(\d\d\d)(\d\d\d)", r"(\1)\2-", string)
    return b


@app.route('/', methods=["GET"])
def home():
    return render_template("PROJECT.html")


@app.route("/predict", methods=["POST"])
def predict():
    f = open('myModel.model', 'rb')
    s = f.read()
    model = pickle.loads(s)
    if request.method == "POST":
        first_name = request.form["First Name"]
        last_name = request.form['Last Name']
        phone_number = request.form['Phone Number']
        gender = request.form['CODE_GENDER']
        car_ownership = request.form['FLAG_OWN_CAR']
        realty_ownership = request.form['FLAG_OWN_REALTY']
        family_members = request.form['CNT_FAM_MEMBERS']
        contract_type = request.form['NAME_CONTRACT_TYPE']
        annual_income = request.form['AMT_INCOME_TOTAL']
        annuity = request.form['AMT_ANNUITY']
        employment_length = request.form['YEARS_EMPLOYED']
        days_registration = request.form['DAYS_REGISTRATION']
        days_id = request.form['DAYS_ID_PUBLISH']
        population = request.form['REGION_POPULATION_RELATIVE']

        input_data = [{
            "First Name": first_name,
            "Last Name": last_name,
            "Phone Number": convert_number(str(phone_number)),
            'CODE_GENDER': gender,
            "FLAG_OWN_CAR": car_ownership,
            "FLAG_OWN_REALTY": realty_ownership,
            'CNT_FAM_MEMBERS': family_members,
            'NAME_CONTRACT_TYPE': contract_type,
            "AMT_INCOME_TOTAL": annual_income,
            'AMT_ANNUITY': annuity,
            'YEARS_EMPLOYED': employment_length,
            'DAYS_REGISTRATION': days_registration,
            'DAYS_ID_PUBLISH': days_id,
            'REGION_POPULATION_RELATIVE': population

        }]

        data = pd.DataFrame(input_data)
        data.to_csv("loan_database.csv", mode='a', header=False)

        predict_input = [{
            'CODE_GENDER': gender,
            "FLAG_OWN_CAR": car_ownership,
            "FLAG_OWN_REALTY": realty_ownership,
            'CNT_FAM_MEMBERS': family_members,
            'NAME_CONTRACT_TYPE': contract_type,
            "AMT_INCOME_TOTAL": annual_income,
            'AMT_ANNUITY': annuity,
            'YEARS_EMPLOYED': employment_length,
            'DAYS_REGISTRATION': days_registration,
            'DAYS_ID_PUBLISH': days_id,
            'REGION_POPULATION_RELATIVE': population
        }]
        prediction_data = pd.DataFrame(predict_input)
        prediction_data.loc[prediction_data.NAME_CONTRACT_TYPE == 'Cash loans', 'NAME_CONTRACT_TYPE'] = 1
        prediction_data.loc[prediction_data.NAME_CONTRACT_TYPE == 'Revolving loans', 'NAME_CONTRACT_TYPE'] = 1
        prediction_data.loc[prediction_data.CODE_GENDER == 'M', 'CODE_GENDER'] = 1
        prediction_data.loc[prediction_data.CODE_GENDER == 'F', 'CODE_GENDER'] = 0
        prediction_data.loc[prediction_data.FLAG_OWN_CAR == 'Y', 'FLAG_OWN_CAR'] = 1
        prediction_data.loc[prediction_data.FLAG_OWN_CAR == 'N', 'FLAG_OWN_CAR'] = 0
        prediction_data.loc[prediction_data.FLAG_OWN_REALTY == 'Y', 'FLAG_OWN_REALTY'] = 1
        prediction_data.loc[prediction_data.FLAG_OWN_REALTY == 'N', 'FLAG_OWN_REALTY'] = 0
        predication_result = model.predict(prediction_data)
        p_result = model.predict_proba(prediction_data)
        if predication_result == 0:
            predication_result = "Granted"
            p = ("%.2f" % p_result[:, 0])
        else:
            predication_result = "Rejected"
            p = ("%.2f" % p_result[:, 1])

        return render_template("RESULTS.html", script=predication_result, probability=p)


if __name__ == '__main__':
    app.run(debug=True)
