import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import joblib

def preprocess_data(input_path, output_dir):
    print(f"--- Starting Preprocessing of {input_path} ---")
    
    # Load data
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return
        
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows.")
    
    # 1. Handling Missing Values
    # Based on analysis, there are no missing values, but we'll keep this for robustness.
    df = df.dropna()
    
    # 2. Encoding Categorical Variables (One-Hot Encoding for 'branch')
    print("Encoding categorical variables...")
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    branch_encoded = ohe.fit_transform(df[['branch']])
    branch_columns = ohe.get_feature_names_out(['branch'])
    branch_df = pd.DataFrame(branch_encoded, columns=branch_columns, index=df.index)
    
    # Combine back
    df_processed = pd.concat([df.drop('branch', axis=1), branch_df], axis=1)
    
    # 3. Define Features and Target
    # We want to keep all columns except 'placed' as features
    target = 'placed'
    X = df_processed.drop(target, axis=1)
    y = df_processed[target]
    
    features = X.columns.tolist()
    print(f"Features: {features}")
    
    # 4. Feature Scaling
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=features, index=df.index)
    
    # 5. Split and Save
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)
    
    # Prepare dataframes for saving
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    full_processed_df = pd.concat([X_scaled_df, y], axis=1)
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Save artifacts
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    joblib.dump(ohe, os.path.join(output_dir, 'encoder.pkl'))
    joblib.dump(features, os.path.join(output_dir, 'features.pkl'))
    
    # Save split files
    full_processed_df.to_csv(os.path.join(output_dir, 'processed_data.csv'), index=False)
    train_df.to_csv(os.path.join(output_dir, 'train_data.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test_data.csv'), index=False)
    
    print(f"Preprocessing complete. Files saved to {output_dir}:")
    print("- processed_data.csv")
    print("- train_data.csv")
    print("- test_data.csv")
    print("- scaler.pkl")
    print("- encoder.pkl")
    print("- features.pkl")
    
    return X_scaled, y, ohe, scaler

if __name__ == "__main__":
    # Get current script path to determine base path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.dirname(current_dir)
    
    input_csv = os.path.join(base_path, 'data', 'student_placement_data.csv')
    output_path = os.path.join(base_path, 'models')
    
    preprocess_data(input_csv, output_path)
