from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('parkinsons_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    input_data = {
        'MDVP:Fo(Hz)': float(request.form['fo']),
        'MDVP:Fhi(Hz)': float(request.form['fhi']),
        'MDVP:Flo(Hz)': float(request.form['flo']),
        'MDVP:Jitter(%)': float(request.form['jitter']),
        'MDVP:Jitter(Abs)': float(request.form['jitter_abs']),
        'MDVP:RAP': float(request.form['rap']),
        'MDVP:PPQ': float(request.form['ppq']),
        'Jitter:DDP': float(request.form['ddp']),
        'MDVP:Shimmer': float(request.form['shimmer']),
        'MDVP:Shimmer(dB)': float(request.form['shimmer_db']),
        'Shimmer:APQ3': float(request.form['apq3']),
        'Shimmer:APQ5': float(request.form['apq5']),
        'MDVP:APQ': float(request.form['apq']),
        'Shimmer:DDA': float(request.form['dda']),
        'NHR': float(request.form['nhr']),
        'HNR': float(request.form['hnr']),
        'RPDE': float(request.form['rpde']),
        'DFA': float(request.form['dfa']),
        'spread1': float(request.form['spread1']),
        'spread2': float(request.form['spread2']),
        'D2': float(request.form['d2']),
        'PPE': float(request.form['ppe'])
    }

    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    # Make prediction
    prediction = model.predict(input_df)
    result = "Likely Parkinson's" if prediction[0] == 1 else "Unlikely Parkinson's"

    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)