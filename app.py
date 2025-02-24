from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the model and scalers
model = joblib.load('random_forest_model.pkl')
X_scaler = joblib.load('X_scaler.pkl')
y_scaler = joblib.load('y_scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    features = [float(request.form['Open']),
                float(request.form['High']),
                float(request.form['Low']),
                float(request.form['Volume']),
                float(request.form['7-day-MA']),
                float(request.form['25-day-MA']),
                float(request.form['Volatility_Index'])]
    
    # Scale the input features
    features_scaled = X_scaler.transform([features])
    
    # Make a prediction
    prediction_scaled = model.predict(features_scaled)
    
    # Inverse transform the prediction
    prediction = y_scaler.inverse_transform(prediction_scaled.reshape(-1, 1))
    
    # Return the prediction to the user
    return render_template('index.html', prediction_text=f'Predicted Bitcoin Price: ${prediction[0][0]:.2f}')

if __name__ == '__main__':
    app.run(debug=True)