import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import pickle
import os

# Load dataset
data = pd.read_csv('insurance.csv')

# Preprocessing function
def preprocess(df):
    df = df.copy()

    # Convert categorical values to numerical
    df['smoker'] = df['smoker'].replace({'yes': 1, 'no': 0})
    df['sex'] = df['sex'].replace({'male': 1, 'female': 0})

    # One-hot encoding for regions
    reg_dummies = pd.get_dummies(df['region'], prefix='region', dtype=int)
    df = pd.concat([df, reg_dummies], axis=1)
    df.drop('region', axis=1, inplace=True)

    X = df.drop('expenses', axis=1)
    y = df['expenses']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.25)

    # Standardize features
    sc = StandardScaler()
    sc.fit(X_train)
    X_train = pd.DataFrame(sc.transform(X_train), columns=X.columns)
    X_test = pd.DataFrame(sc.transform(X_test), columns=X.columns)
    
    return X_train, X_test, y_train, y_test, sc, X.columns

# Train the model and save it
def train_model():
    X_train, X_test, y_train, y_test, sc, columns = preprocess(data)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save model, scaler, and columns
    with open('best_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    
    with open('scaler.pkl', 'wb') as scaler_file:
        pickle.dump(sc, scaler_file)

    with open('columns.pkl', 'wb') as columns_file:
        pickle.dump(columns, columns_file)

    return model, sc, columns

# Load the model, scaler, and columns
def load_model():
    if os.path.exists('best_model.pkl') and os.path.exists('scaler.pkl') and os.path.exists('columns.pkl'):
        with open('best_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        
        with open('scaler.pkl', 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
        
        with open('columns.pkl', 'rb') as columns_file:
            columns = pickle.load(columns_file)
        
        return model, scaler, columns
    else:
        return None, None, None

# Preprocess the input data before prediction
def preprocess_input(input_data, columns):
    input_df = pd.DataFrame([input_data], columns=columns)

    # Convert categorical columns ('sex', 'region', etc.) like during training
    input_df['sex'] = input_df['sex'].replace({'male': 1, 'female': 0})
    input_df['smoker'] = input_df['smoker'].replace({'yes': 1, 'no': 0})

    # Check if 'region' column exists in the input data
    if 'region' in input_df.columns:
        reg_dummies = pd.get_dummies(input_df['region'], prefix='region', dtype=int)
        input_df = pd.concat([input_df, reg_dummies], axis=1)
        input_df.drop('region', axis=1, inplace=True)
    else:
        # If 'region' is missing, assume a default value or handle the missing data
        input_df['region_northeast'] = 0
        input_df['region_northwest'] = 0
        input_df['region_southeast'] = 0
        input_df['region_southwest'] = 0

    # Ensure the input columns match the model training columns
    missing_cols = set(columns) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0  # Adding missing columns with 0 values
    
    input_df = input_df[columns]

    return input_df

# Make a prediction
def predict(input_data):
    model, scaler, columns = load_model()

    if model is None:
        return "Model is not trained yet. Please train the model first by visiting the /train route."

    # Preprocess the input data
    input_df = preprocess_input(input_data, columns)
    input_df_scaled = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(input_df_scaled)

    return prediction[0]

# Route to train the model
def train_route():
    model, sc, columns = train_model()
    return "Model trained successfully!"

# Example usage for testing
if __name__ == '__main__':
    print("Training the model...")
    train_route()  # Train the model
    print("Model training complete!")
