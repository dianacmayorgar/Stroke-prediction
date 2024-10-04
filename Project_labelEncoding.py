import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(filepath):
    """Load the dataset from a CSV file."""
    return pd.read_csv(filepath)

def explore_data(df):
    """Perform basic exploration of the dataset."""
    print(df.info())
    print(df.describe())
    print("Missing values per column:")
    print(df.isnull().sum())

def handle_missing_values(df):
    """Handle missing values by filling or removing them."""
    # Fill missing BMI values with the mean BMI grouped by age
    df['bmi'] = df.groupby('age')['bmi'].transform(lambda x: x.fillna(x.mean()))

    # Drop rows where 'smoking_status' is missing
    df.dropna(subset=['smoking_status'], inplace=True)
    
    return df

def check_data_quality(df):
    """Check for errors or inconsistencies in the data."""
    print("Unique values in 'gender':", df['gender'].unique())
    print("Value counts for 'gender':")
    print(df['gender'].value_counts())

def visualize_distributions(df):
    """Visualize distributions for key variables."""
    sns.histplot(df['bmi'])
    plt.title("BMI Distribution")
    plt.show()

def remove_duplicates(df):
    """Remove duplicate records if any."""
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

def main():
    filepath = 'train_2v.csv'  # Adjust the file path as needed
    df = load_data(filepath)

    # Step 1: Explore the data
    explore_data(df)
    
    # Step 2: Handle missing values
    df = handle_missing_values(df)

    # Step 3: Check for inconsistencies and data quality
    check_data_quality(df)
    
    # Step 4: Visualize distributions
    visualize_distributions(df)
    
    # Step 5: Remove duplicates
    remove_duplicates(df)

if __name__ == "__main__":
    main()
