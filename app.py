from flask import Flask, request, render_template
import joblib
import numpy as np
import os

# Load the trained model using joblib
# Use relative path to ensure it works across environments (local and deployed)
model_path = os.path.join(os.path.dirname(__file__), 'best_model1.pkl')  # Assuming the model is in the same directory as app.py

# Ensure the model file exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at: {model_path}")

model = joblib.load(model_path)

app = Flask(__name__)

@app.route('/')
def home():
    # Render the home page with the input form
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract data from the form
        patient_name = request.form['patientName']
        patient_age = int(request.form['patientAge'])
        donor_age = int(request.form['donorAge'])
        time_to_agvhd = int(request.form['timeToAGvHD'])
        recipient_abo = int(request.form['recipientABO'])
        gender = request.form['patientGender']
        disease = request.form['disease']
        
        # Prepare features for the model (following the same format as used during training)
        features = [
            patient_age,
            donor_age,
            time_to_agvhd,
            recipient_abo,
            1 if gender == 'male' else 0,  # Binary encoding for gender
            1 if disease == 'acute' else 0  # Binary encoding for disease
        ]
        
        final_features = [np.array(features)]  # Convert the features into a NumPy array
        
        # Make prediction using the trained model
        prediction = model.predict(final_features)
        
        # Interpret the prediction result
        output = 'Survived' if prediction[0] == 1 else 'Not Survived'
        
        # Render the result on the web page
        return render_template('index.html', prediction_text=f'Prediction: {output}')
    
    except Exception as e:
        # Handle errors and return the error message
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
