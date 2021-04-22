# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 20:56:11 2021

@author: luole
"""
from flask import Flask,render_template_string, request
from wtforms import Form, validators,TextField
import string

app = Flask(__name__)

def compute(original_str):
    bad_chars = string.whitespace =string.punctuation
    modified_str = original_str.lower()
    for char in modified_str:
        if char in bad_chars:
            modified_str = modified_str.replace(char,'')
    if modified_str == modified_str[::-1]:
        return "It is a palindrome"
    else:
        return "It is not a palindrome"
    
@app.route('/')
def page():
    return '''
<html>
    <head>
        <title> FAO Financials </title>
    </head>

    <body>
        <h1 style="color:orange;"> Welcome to FAO Financials</h1>
        <h2>Loan Application Page</h2>
        <h3>Personal Information</h3>
        <from method = post action="">
            Hello Applicant!
        <h3>Other Information</h3>
        <br>
        Enter a phone number:
        <br>
            
        <br>
          <input type = submit value*calculate>
        </from>
  
    </body>  
</html>
'''
class InputForm(Form):
    text_field = TextField(validators=[validators.InputRequired()])
@app.route('/',methods=['GET','POST'])

def index():
    palindrome_result = None
    form = InputForm(request.form)
    if request.method == 'POST' and form.validators():
        input_val = form.text_field.data
        palindrome_result = compute(input_val)
    return render_template_string(page,template_form = form,reult = palindrome_result)

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=7777)
    