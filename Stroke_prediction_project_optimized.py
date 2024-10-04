import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import save_model

# Data loading
def load_data(filepath):
    """Load the dataset from a CSV file."""
    df = pd.read_csv(filepath)
    return df

# Data exploration and initial statistics
def explore_data(df):
    """Perform basic exploration of the dataset."""
    print(df.info())
    print(df.describe())
    print("Missing values per column:")
    print(df.isnull().sum())

# Handle missing values
def handle_missing_values(df):
    """Handle missing values efficiently."""
    df['bmi'] = df.groupby('age')['bmi'].transform(lambda x: x.fillna(x.mean()))
    df.dropna(subset=['smoking_status'], inplace=True)
    return df

# Visualize key distributions
def visualize_distributions(df):
    sns.histplot(df['bmi'])
    plt.title("BMI Distribution")
    plt.show()

# Feature scaling
def scale_features(X_train, X_test):
    """Standardize the feature set."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# Train and evaluate a model
def train_and_evaluate_model(X_train, y_train, X_test, y_test):
    """Train a RandomForestClassifier and evaluate its performance."""
    model = RandomForestClassifier(random_state=42)
    
    # Hyperparameter tuning with GridSearchCV
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None]
    }
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Main pipeline
def main():
    filepath = 'data/train_2v.csv'
    df = load_data(filepath)
    
    # Step 1: Explore the data
    explore_data(df)
    
    # Step 2: Handle missing values
    df = handle_missing_values(df)
    
    # Step 3: Visualize distributions
    visualize_distributions(df)
    
    # Step 4: Prepare data for machine learning
    X = df.drop('stroke', axis=1)
    y = df['stroke']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Step 5: Scale features
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    
    # Step 6: Train and evaluate model
    train_and_evaluate_model(X_train_scaled, y_train, X_test_scaled, y_test)

if __name__ == "__main__":
    main()
