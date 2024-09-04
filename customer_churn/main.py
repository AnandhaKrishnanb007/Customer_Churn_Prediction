from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the scaler and model
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('decision_tree_model.pkl', 'rb') as file:
    model = pickle.load(file)


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None
    probability = None

    if request.method == 'POST':
        # Collect input data
        data = [
            int(request.form['PreferredLoginDevice']),
            int(request.form['CityTier']),
            float(request.form['WarehouseToHome']),
            int(request.form['PreferredPaymentMode']),
            int(request.form['Gender']),
            int(request.form['NumberOfDeviceRegistered']),
            int(request.form['PreferedOrderCat']),
            int(request.form['SatisfactionScore']),
            int(request.form['MaritalStatus']),
            int(request.form['NumberOfAddress']),
            int(request.form['Complain']),
            float(request.form['CouponUsed']),
            float(request.form['OrderCount']),
            float(request.form['DaySinceLastOrder']),
            float(request.form['CashbackAmount']),
        ]

        # Convert to numpy array and reshape
        data = np.array(data).reshape(1, -1)

        # Scale the data
        data = scaler.transform(data)

        # Make prediction
        prediction = model.predict(data)[0]
        probability = model.predict_proba(data)[0][1]

    return render_template('index.html', prediction=prediction, probability=probability)



if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
