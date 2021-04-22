# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 17:48:20 2021

@author: luole
"""

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
