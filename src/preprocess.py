import pandas as pd
import numpy as np
from sklearn import preprocessing

def clean_data(csv_path):
    """
    Loads raw data, removes non-predictive features, standardizes 
    column naming, and removes rows with missing values.
    
    Returns:
        pd.DataFrame: Cleaned dataset ready for feature selection.
    """
    # Load Dataset
    disease_df = pd.read_csv(csv_path)
    
    # Drop irrelevant features
    disease_df.drop(['education'], axis=1, inplace=True)
    
    # Handle Missing Values (Listwise deletion)
    disease_df.dropna(axis=0, inplace=True)
    
    return disease_df


def get_features_and_target(df):
    """
    Extracts specific predictors and the target variable, then 
    standardizes features to have a mean of 0 and variance of 1.
    
    Returns:
        tuple: (scaled_X, y)
    """
    features = ['age', 'male', 'cigsPerDay', 'totChol', 'sysBP', 'glucose']
    X = np.asarray(df[features])
    y = np.asarray(df['TenYearCHD'])
    
    # Standardization (Scaling)
    # Essential for Logistic Regression to ensure coefficients are comparable
    scaler = preprocessing.StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y