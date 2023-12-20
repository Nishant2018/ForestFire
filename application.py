import pickle
from flask import Flask,request,jsonify,render_template
from flask_cors import CORS,cross_origin
import requests
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import joblib

application = Flask(__name__)
app=application
CORS(app)

## import linar regresor model and standard scaler pickle
linear_model=joblib.load('linreg.pkl')
standard_scaler=joblib.load('scaler.pkl')

@app.route('/',methods=['GET','POST'])
@cross_origin()
def predict_datapoint():
    if request.method=='POST':
        Temperature=float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        BUI = float(request.form.get('BUI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        new_data_scaled=standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,BUI,Classes,Region]])
        result=linear_model.predict(new_data_scaled)

        return render_template('home.html',result=result[0])

    else:
        return render_template('home.html')


if __name__=="__main__":
    app.run(host="0.0.0.0",port=8000)
    