from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np
import pickle
import json
import os

app = Flask(__name__)

# Load model, label encoder, and features
model = pickle.load(open("model/churn_model.pkl", "rb"))
le = pickle.load(open("model/label_encoder.pkl", "rb"))
with open("model/features.json", "r") as f:
    feature_cols = json.load(f)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    uploaded_file = request.files['file']
    if not uploaded_file:
        return render_template("index.html", error="No file uploaded")

    df = pd.read_csv(uploaded_file)

    for col in df.select_dtypes(include='object').columns:
        if col != 'Churn' and col in feature_cols:
            df[col] = le.fit_transform(df[col])

    input_df = df[feature_cols].copy()
    df['Predicted Churn'] = model.predict(input_df)
    df['Churn Probability'] = model.predict_proba(input_df)[:, 1]
    df['Recommendation'] = np.where(df['Predicted Churn'] == 1,
                                    '📉 Offer retention benefit',
                                    '🤝 Maintain engagement')

    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        df['Prediction Status'] = np.where(
            df['Predicted Churn'] == df['Churn'], '✅ Correct', '❌ Incorrect'
        )

    output_path = "static/results.csv"
    df.to_csv(output_path, index=False)

    return render_template(
        "index.html",
        tables=[df.head(10).to_html(classes='table table-bordered', index=False)],
        download=True
    )

if __name__ == '__main__':
    app.run(debug=True)
