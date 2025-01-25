from flask import Flask, render_template, request
from model import predict, train_model
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', prediction=None)

@app.route('/train', methods=['GET'])
def train():
    model, sc, columns = train_model()
    return "Model has been trained successfully!"

@app.route('/predict', methods=['POST'])
def make_prediction():
    # Retrieve the form data
    age = float(request.form['age'])
    sex = request.form['sex']
    bmi = float(request.form['bmi'])
    children = int(request.form['children'])
    smoker = request.form['smoker']
    region = request.form['region']
    
    # Prepare input data as a dictionary
    input_data = {
        'age': age,
        'sex': sex,
        'bmi': bmi,
        'children': children,
        'smoker': smoker,
        'region': region
    }
    
    # Predict the result
    prediction = predict(input_data)
    
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
