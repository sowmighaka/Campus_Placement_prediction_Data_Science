from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__, 
            template_folder='../frontend/templates', 
            static_folder='../frontend/static')

# Load artifacts
base_path = os.path.dirname(os.path.abspath(__file__))
models_path = os.path.join(base_path, 'models')

# Safely load models and tools
try:
    models = joblib.load(os.path.join(models_path, 'models.pkl'))
    scaler = joblib.load(os.path.join(models_path, 'scaler.pkl'))
    ohe = joblib.load(os.path.join(models_path, 'encoder.pkl'))
    features_list = joblib.load(os.path.join(models_path, 'features.pkl'))
    print("Artifacts loaded successfully.")
except Exception as e:
    print(f"Error loading artifacts: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # 1. Create initial dataframe with numeric features
        input_dict = {
            'ssc_percentage': [float(data['ssc_p'])],
            'hsc_percentage': [float(data['hsc_p'])],
            'cgpa': [float(data['cgpa'])],
            'backlogs': [int(data['backlogs'])],
            'programming_skill': [int(data['prog_skill'])],
            'projects': [int(data['projects'])],
            'internship': [1 if data['internship'] == "Yes" else 0],
            'aptitude_score': [float(data['aptitude'])],
            'communication_skills': [int(data['comm_skill'])],
            'mock_interviews': [int(data['mock_interviews'])]
        }
        input_df = pd.DataFrame(input_dict)
        
        # 2. Handle Branch Encoding (One-Hot)
        branch = data['branch']
        branch_encoded = ohe.transform([[branch]])
        branch_columns = ohe.get_feature_names_out(['branch'])
        branch_df = pd.DataFrame(branch_encoded, columns=branch_columns)
        
        # 3. Combine numeric and encoded features
        final_input_df = pd.concat([input_df, branch_df], axis=1)
        
        # 4. Ensure feature order matches the training data
        final_input_df = final_input_df[features_list]
        
        # 5. Scaling
        input_scaled = scaler.transform(final_input_df)
        
        # 6. Prediction (using Random Forest as default)
        model = models['Random Forest']
        prediction = int(model.predict(input_scaled)[0])
        probability = float(model.predict_proba(input_scaled)[0][1])
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'probability': probability
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=False)
