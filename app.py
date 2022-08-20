from flask import Flask, render_template, jsonify, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

@app.route('/')
@app.route('/inicio')

def index():
    return render_template('/templates/index.html')

loaded_model = pickle.load(open('coffee_model.pkl', 'rb')) # rb: read binary

@app.route('/<string:country>/<string:variety>/<float:aroma>/<float:aftertaste>/<float:acidity>/<float:body>/<float:balance>/<float:moisture>')
def result(country, variety, aroma, aftertaste, acidity, body, balance, moisture):
    cols = ['country_of_origin', 'variety', 'aroma', 'aftertaste', 'acidity', 'body', 'balance', 'moisture']
    data = [country, variety, aroma, aftertaste, acidity, body, balance, moisture]
    posted = pd.DataFrame(np.array(data).reshape(1,8), columns=cols)

    result = loaded_model.predict(posted)
    text_result = result.tolist()[0]
    
    if text_result == 'Yes':
        return jsonify(message='Si es un cafe de especialiad'), 200
    else:
        return jsonify(message='No es un cafe de especialidad'), 200 
    

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)