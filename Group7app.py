# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 13:19:19 2020

@author: Ramya
"""
 
import os
import pandas as pd

import json

from flask import Flask
from flask_restful import Api, Resource, request
import joblib


port = int(os.getenv('PORT', '5000'))

app = Flask(__name__)
api = Api(app)

# argument parsing
#parser = reqparse.RequestParser()
#parser.add_argument('query')


class ForecastTimeSeriesEts(Resource):
    def get(self):
        model_path = 'lib/models/timeSeriesModel.pkl'
        with open(model_path, 'rb') as f:
            ets_model = joblib.load(f)
                
        # get the query parameters
        steps = int(request.args.get('steps'))
        
        # make a prediction
        pred_uc = ets_model.forecast(steps)
       
        # create JSON object
        output = pred_uc.to_json(orient='table')
        output=json.loads(output)
        
        return output
    
class ForecastTimeSeriesSarima(Resource):
    def get(self):
        model_path = 'lib/models/timeSeriesModel.pkl'
        with open(model_path, 'rb') as f:
            sarima_model = joblib.load(f)
                
        # get the query parameters
        steps = int(request.args.get('steps'))
        
        # make a prediction
        pred_uc = sarima_model.get_forecast(steps=steps)
        pred_ci = pred_uc.conf_int(alpha=.05)
        pred_ci['Prediction'] = pred_uc.predicted_mean
        pred_ci.columns=['Lower','Upper','Prediction']
        
        print(pred_ci)
        print(type(pred_ci))
       
        # create JSON object
        output = pred_ci.to_json(orient='table')
        output=json.loads(output)
        
        return output
  
class PredictRegression(Resource):
    def get(self):
        model_path = 'models/group7regressionmodel.pkl'
        with open(model_path, 'rb') as f:
            regression_model = joblib.load(f)
                
        # get the query parameters
        #param = request.args.get('param')
        params = request.args

        independents=pd.DataFrame(params,index=[0])
              
        # make a prediction
        predictions = regression_model.predict(independents)
       
        # create JSON object
        #pd.DataFrame(predictions, columns=['Prediction'])
        output = pd.DataFrame(predictions, columns=['Prediction']).to_json(orient='table')
        output=json.loads(output)
        
        return output
    
    def post(self):
        model_path = 'lib/models/regressionModel.pkl'
        with open(model_path, 'rb') as f:
            regression_model = joblib.load(f)
                
        # get the form data
        params = request.form

        independents=pd.DataFrame(params,index=[0])
              
        # make a prediction
        predictions = regression_model.predict(independents)
       
        # create JSON object
        #pd.DataFrame(predictions, columns=['Prediction'])
        output = pd.DataFrame(predictions, columns=['Prediction']).to_json(orient='table')
        output=json.loads(output)
        
        return output

class PredictClassification(Resource):
    def get(self):
        model_path = 'lib/models/classificationModel.pkl'
        with open(model_path, 'rb') as f:
            classification_model = joblib.load(f)
                
       # get the query parameters
        params = request.args

        inputs=pd.DataFrame(params,index=[0])
              
        # make a prediction
        predictions = classification_model.predict(inputs)
       
        # create JSON object
        #pd.DataFrame(predictions, columns=['Prediction'])
        output = pd.DataFrame(predictions, columns=['Prediction']).to_json(orient='table')
        output=json.loads(output)
        
        return output
    
    def post(self):
        model_path = 'lib/models/classificationModel.pkl'
        with open(model_path, 'rb') as f:
            classification_model = joblib.load(f)
                
       # get the form data
        params = request.form

        inputs=pd.DataFrame(params,index=[0])
              
        # make a prediction
        predictions = classification_model.predict(inputs)
       
        # create JSON object
        #pd.DataFrame(predictions, columns=['Prediction'])
        output = pd.DataFrame(predictions, columns=['Prediction']).to_json(orient='table')
        output=json.loads(output)
        
        return output



# Setup the Api resource routing here
# Route the URL to the resource
api.add_resource(ForecastTimeSeriesEts, '/time_ets')
api.add_resource(ForecastTimeSeriesSarima, '/time_sarima')
api.add_resource(PredictRegression, '/regression')
api.add_resource(PredictClassification, '/classification')


if __name__ == '__main__':
    #app.run(debug=True)
    app.run(host='0.0.0.0', port=port)