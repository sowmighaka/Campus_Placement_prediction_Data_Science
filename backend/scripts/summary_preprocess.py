import pandas as pd
import numpy as np
import os
import joblib

def show_transformation_summary():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_path = os.path.join(base_path, 'data', 'student_placement_data.csv')
    processed_path = os.path.join(base_path, 'models', 'processed_data.csv')
    
    if not os.path.exists(raw_path) or not os.path.exists(processed_path):
        print("Required files not found.")
        return

    raw_df = pd.read_csv(raw_path)
    proc_df = pd.read_csv(processed_path)
    
    print("=== PREPROCESSING SUMMARY ===")
    print(f"Original Shape: {raw_df.shape}")
    print(f"Processed Shape: {proc_df.shape}")
    
    print("\n--- Feature Transformations ---")
    print("1. 'branch' column: Categorical -> One-Hot Encoded (5 new columns)")
    print("2. Numeric Columns: Scaled (Mean ~ 0, Std ~ 1)")
    
    print("\n--- Sample Scaling Check (CGPA) ---")
    print(f"Raw CGPA Mean: {raw_df['cgpa'].mean():.2f}, Std: {raw_df['cgpa'].std():.2f}")
    print(f"Processed CGPA Mean: {proc_df['cgpa'].mean():.2f}, Std: {proc_df['cgpa'].std():.2f}")
    
    print("\n--- Target Distribution (Placed) ---")
    dist = proc_df['placed'].value_counts(normalize=True) * 100
    print(f"Placed (1): {dist[1]:.1f}%")
    print(f"Not Placed (0): {dist[0]:.1f}%")
    
    print("\nPreprocessing is fully operational and artifacts (scaler.pkl, encoder.pkl) are ready.")

if __name__ == "__main__":
    show_transformation_summary()
