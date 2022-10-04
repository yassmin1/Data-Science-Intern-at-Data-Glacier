from webbrowser import get
import numpy as np
import pandas as pd
import pickle
from flask import Flask,request,render_template,jsonify

app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))
@app.route('/') # take to the out page
def home():
    return render_template('index.html')
@app.route('/predict',methods=['POST','GET'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    if output == 1:
        output_text='Diabetic'
    elif output == 0:
        output_text='Not Diabetic'    

    return render_template('index.html',prediction_text=f'You are {output_text}')
if __name__ == "__main__":
    app.run(debug=True)  