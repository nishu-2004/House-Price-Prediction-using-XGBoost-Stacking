from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import os

# Print the templates folder path
print("Templates Folder Path: ", os.path.join(os.getcwd(), 'templates'))

# Load the model
model = joblib.load('house_price_model.pkl')

# Load the original data to extract feature columns
from sklearn.datasets import fetch_california_housing
california_housing = fetch_california_housing(as_frame=True)
df = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
df['MedHouseVal'] = california_housing.target

# Columns to remove (matching the training preprocessing step)
columns_to_remove = ['AveBedrms', 'Longitude']
X_columns = df.drop(columns=columns_to_remove).drop(columns='MedHouseVal').columns  # Features used during training

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # Serve the HTML file

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract the input data from the form
        feature_values = [
            float(request.form['MedInc']),
            float(request.form['HouseAge']),
            float(request.form['AveRooms']),
            float(request.form['Population']),
            float(request.form['AveOccup']),
            float(request.form['Latitude'])
        ]

        # Convert the input data into a pandas DataFrame with the correct column names
        input_data = pd.DataFrame([feature_values], columns=X_columns)

        # Make the prediction
        prediction = model.predict(input_data)

        # Return the prediction as a response
        return f"Predicted House Price: ${prediction[0]:.2f}"

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)