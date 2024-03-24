from flask import Flask,render_template,url_for,request
import pickle
import numpy as np
import joblib

app=Flask(__name__)

model_path = './model.plk'
model=joblib.load(
    open(model_path, 'rb'))

@app.route('/')

def home():
    return render_template('index.html')

@app.route('/result',methods=['POST'])
def predict():
    # Getting the data from the form
    N = (request.form['N'])
    P = (request.form['P'])
    K = (request.form['K'])
    humidity =(request.form['humidity'])
    rainfall = (request.form['rainfall'])

    
    querry=np.array([[N, P, K, humidity, rainfall]])
    prediction = model.predict(querry)

    
    return render_template('result.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
