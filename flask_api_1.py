from typing import Text
from flask import Flask,request, url_for, redirect, render_template, jsonify
import numpy as np
import pandas as pd
import pickle



app = Flask(__name__)

# load the pickle file and open it in read_byte mode

pickle_in = open("model.pkl","rb")
model = pickle.load(pickle_in)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=["POST"])
def predict_note_authentication():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features, dtype=float)]
    prediction = model.predict(final_features)

    # Round-off the output up to 3 decimal places
    output = round(prediction[0],3)

    return render_template('index.html', text1=' Predicted Prices are : ', predicted_value= '{}'.format(output), text2= ' U$D Only. ')


if __name__ == '__main__':
    app.run(debug=True)
