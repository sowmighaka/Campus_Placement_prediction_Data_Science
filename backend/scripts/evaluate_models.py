import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_models(models_path, test_data_path, output_dir):
    print(f"--- Starting Model Evaluation ---")
    
    # Load models
    models = joblib.load(os.path.join(models_path, 'models.pkl'))
    
    # Load test data
    df_test = pd.read_csv(test_data_path)
    X_test = df_test.drop('placed', axis=1)
    y_test = df_test['placed']
    
    results = []
    
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        print(f"Accuracy: {acc:.4f}")
        print("Classification Report:")
        print(report)
        
        results.append({
            'Model': name,
            'Accuracy': acc
        })
        
        # Plot Confusion Matrix
        plt.figure(figsize=(6, 4))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        # Save plot
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(os.path.join(output_dir, f'cm_{name.lower().replace(" ", "_")}.png'))
        plt.close()

    # Save summary report
    summary_df = pd.DataFrame(results)
    summary_df.to_csv(os.path.join(output_dir, 'evaluation_summary.csv'), index=False)
    print(f"\nEvaluation complete. Reports and plots saved to {output_dir}")

if __name__ == "__main__":
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_path, 'models')
    test_csv = os.path.join(base_path, 'models', 'test_data.csv')
    output_path = os.path.join(base_path, 'models', 'evaluation_reports')
    evaluate_models(models_dir, test_csv, output_path)
