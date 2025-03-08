from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import os
print("Templates Folder Path: ", os.path.join(os.getcwd(), 'templates'))

model = joblib.load('house_price_model.pkl')

from sklearn.datasets import fetch_california_housing
california_housing = fetch_california_housing(as_frame=True)
df = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
df['MedHouseVal'] = california_housing.target

columns_to_remove = ['AveBedrms', 'Longitude']
X_columns = df.drop(columns=columns_to_remove).drop(columns='MedHouseVal').columns  

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

        input_data = pd.DataFrame([feature_values], columns=X_columns)

        prediction = model.predict(input_data)

        return f"Predicted House Price: ${prediction[0]:.2f}"

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
