import pandas as pd
import numpy as np
import os

def generate_placement_data(n_samples=1000):
    np.random.seed(42)
    
    # Features
    ssc_p = np.random.uniform(50, 95, n_samples)
    hsc_p = np.random.uniform(50, 95, n_samples)
    cgpa = np.random.uniform(5.0, 10.0, n_samples)
    backlogs = np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.7, 0.15, 0.08, 0.05, 0.02])
    branch = np.random.choice(['CSE', 'ECE', 'ME', 'CE', 'EE'], n_samples)
    prog_skill = np.random.randint(1, 11, n_samples)
    projects = np.random.randint(0, 6, n_samples)
    internship = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
    aptitude = np.random.uniform(40, 100, n_samples)
    comm_skill = np.random.randint(1, 11, n_samples)
    mock_interviews = np.random.randint(0, 11, n_samples)
    
    # Calculate Placement Probability based on features
    # Base probability
    prob = (
        0.3 * (cgpa - 5) / 5 +
        0.15 * (ssc_p - 50) / 45 +
        0.15 * (hsc_p - 50) / 45 +
        0.2 * (prog_skill / 10) +
        0.1 * (aptitude - 40) / 60 +
        0.1 * (projects / 5) +
        0.1 * internship -
        0.2 * (backlogs / 4)
    )
    
    # Add some randomness
    prob += np.random.normal(0, 0.1, n_samples)
    
    # Clip and convert to binary
    prob = np.clip(prob, 0, 1)
    placed = (prob > 0.5).astype(int)
    
    df = pd.DataFrame({
        'ssc_percentage': ssc_p,
        'hsc_percentage': hsc_p,
        'cgpa': cgpa,
        'backlogs': backlogs,
        'branch': branch,
        'programming_skill': prog_skill,
        'projects': projects,
        'internship': internship,
        'aptitude_score': aptitude,
        'communication_skills': comm_skill,
        'mock_interviews': mock_interviews,
        'placed': placed
    })
    
    return df

if __name__ == "__main__":
    df = generate_placement_data()
    output_path = os.path.join(os.path.dirname(__file__), "student_placement_data.csv")
    df.to_csv(output_path, index=False)
    print(f"Dataset generated with {len(df)} samples at {output_path}")
