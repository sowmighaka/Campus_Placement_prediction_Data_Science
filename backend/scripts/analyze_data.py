import pandas as pd

df = pd.read_csv(r'c:\Users\sowmi\OneDrive\Documents\DATA_SCIENCE_PROJECT\backend\data\student_placement_data.csv')
print("--- Info ---")
print(df.info())
print("\n--- Descriptive Statistics ---")
print(df.describe())
print("\n--- Unique values for 'branch' ---")
print(df['branch'].unique())
print("\n--- Missing values ---")
print(df.isnull().sum())
