import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import joblib

def train_models(data_path, output_dir):
    print(f"--- Starting Model Training using {data_path} ---")
    
    # Load processed data
    df = pd.read_csv(data_path)
    X = df.drop('placed', axis=1)
    y = df['placed']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(n_estimators=100)
    }
    
    trained_models = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        print(f"{name} Accuracy: {accuracy:.4f}")
        trained_models[name] = model
        
    # Save the dictionary of models
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    joblib.dump(trained_models, os.path.join(output_dir, 'models.pkl'))
    print(f"Model training complete. Saved models to {output_dir}/models.pkl")

if __name__ == "__main__":
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_csv = os.path.join(base_path, 'models', 'processed_data.csv')
    output_path = os.path.join(base_path, 'models')
    train_models(data_csv, output_path)
