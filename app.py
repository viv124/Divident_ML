from flask import Flask, render_template, request, redirect, url_for, flash, send_file
import joblib
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import os
from io import BytesIO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'Uploads'
app.config['OUTPUT_FOLDER'] = 'FilteredOutput'

# Load the trained model and vectorizer outside of the route
loaded_model = joblib.load('model.joblib')
loaded_vectorizer = joblib.load('vectorizer.joblib')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        description_col = int(request.form.get("description"))
        ref_no_col = int(request.form.get("ref_no"))
        credit_col = int(request.form.get("credit"))

        if file:
            df = process_uploaded_file(file, description_col, ref_no_col)
            if df is not None:
                input_data = df['FileData']
                input_data_vectorized = loaded_vectorizer.transform(input_data)
                predictions = loaded_model.predict(input_data_vectorized)
                df['Predictions'] = predictions

                # Convert 'Credit' column to numeric, handling non-numeric values gracefully
                df['Credit'] = pd.to_numeric(df.iloc[:, credit_col], errors='coerce')

                # Filter rows where predictions are 1
                predicted_1 = df[df['Predictions'] == 1]

                # Save the filtered data to 'filtered_data.xlsx' in the 'FilteredOutput' folder
                filtered_filename = os.path.join(app.config['OUTPUT_FOLDER'], 'filtered_data.xlsx')
                predicted_1.to_excel(filtered_filename, index=False)

                # Save the updated DataFrame back to the original file
                df.to_excel(os.path.join(app.config['UPLOAD_FOLDER'], file.filename), index=False)

                total_credit_payment = predicted_1['Credit'].sum()

                return render_template('result.html', predicted_1=predicted_1, total_credit_payment=total_credit_payment)

            else:
                flash('Uploaded file is missing "Description" or "Ref_No" columns.')
                return redirect(url_for('index'))
        else:
            flash('No file uploaded.')
            return redirect(url_for('index'))

@app.route('/download')
def download():
    filtered_filename = os.path.join(app.config['OUTPUT_FOLDER'], 'filtered_data.xlsx')
    return send_file(
        filtered_filename,
        as_attachment=True,
        download_name='filtered_data.xlsx',
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

def process_uploaded_file(file, description_col, ref_no_col):
    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        df = pd.read_excel(filename)
        if 'Description' in df.columns and 'Ref_No' in df.columns:
            df['FileData'] = df.iloc[:, description_col].astype(str) + ' ' + df.iloc[:, ref_no_col].astype(str)
            return df
    return None

if __name__ == '__main__':
    app.run(debug=True)
