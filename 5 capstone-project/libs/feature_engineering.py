import ast
import pandas as pd
import numpy as np
from typing import Dict
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def max_scaling(df, field_name):
    new_column_name = f'max_scaling_{field_name}'
    max_value = df[field_name].max()
    df[new_column_name] = df[field_name] / max_value
    return df


def min_scaling(df, field_name):
    new_column_name = f'min_scaling_{field_name}'
    min_value = df[field_name][df[field_name] != 0].min()
    df[new_column_name] = np.where(df[field_name] == 0, 0, min_value / df[field_name])
    df[new_column_name] = df[new_column_name].fillna(0)
    return df


def score_portfolio(portfolio_df):
    
    portfolio_df = max_scaling(portfolio_df, "reward")
    portfolio_df = min_scaling(portfolio_df, "difficulty")
    portfolio_df = max_scaling(portfolio_df, "duration")
    
    portfolio_df["score"] = (
        portfolio_df["max_scaling_reward"]*0.4 
        + portfolio_df["max_scaling_duration"]*0.2 
        + portfolio_df["min_scaling_difficulty"]*0.4
    )*100
    
    return portfolio_df


def generate_profile_kmeans_clusters(df, n_clusters=3, random_state=42, plot=True):    
    # Select relevant features
    numerical_features = ['age', 'income']
    categorical_features = ['gender']
    
    # Create a copy of the input dataframe
    X = df.copy()
        
    # One-hot encode the gender column
    gender_encoded = pd.get_dummies(X['gender'], prefix='gender', dummy_na=True)
    
    # Combine numerical features and encoded gender
    X_combined = pd.concat([X[numerical_features], gender_encoded], axis=1)
    
    # Create a pipeline that includes imputation and scaling
    preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    # Preprocess the data
    X_preprocessed = preprocessor.fit_transform(X_combined)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    cluster_labels = kmeans.fit_predict(X_preprocessed)
    
    # Add cluster labels to the original dataframe
    df_with_clusters = df.copy()
    df_with_clusters['customer_type'] = cluster_labels
    
    if plot:
        # Visualize clusters (using first three features)
        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(X_preprocessed[:, 0], X_preprocessed[:, 1], X_preprocessed[:, 2], 
                             c=cluster_labels, cmap='viridis')
        ax.set_xlabel('Age (scaled)')
        ax.set_ylabel('Income (scaled)')
        ax.set_zlabel('Years since joining (scaled)')
        plt.title('3D visualization of clusters')
        plt.colorbar(scatter)
        plt.show()

        # Elbow method
        inertias = []
        range_n_clusters = range(1, 11)
        for n in range_n_clusters:
            kmeans = KMeans(n_clusters=n, random_state=random_state)
            kmeans.fit(X_preprocessed)
            inertias.append(kmeans.inertia_)

        plt.figure(figsize=(10, 6))
        plt.plot(range_n_clusters, inertias, marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('Inertia')
        plt.title('Elbow Method for Optimal k')
        plt.show()
    
    return df_with_clusters


def replace_with_quintile_labels(df):
    # Create a copy to avoid modifying the original DataFrame
    df_quintiles = df.copy()
    
    # Iterate over each numerical column in the DataFrame
    for column in df_quintiles.select_dtypes(include=['float64', 'int64']).columns:
        # Calculate the quintiles
        Q1 = df_quintiles[column].quantile(0.20)
        Q2 = df_quintiles[column].quantile(0.40)
        Q3 = df_quintiles[column].quantile(0.60)
        Q4 = df_quintiles[column].quantile(0.80)
        
        # Replace the values with quintile labels
        df_quintiles[column] = pd.cut(df_quintiles[column],
                                       bins=[-float('inf'), Q1, Q2, Q3, Q4, float('inf')],
                                       labels=['E', 'D', 'C', 'B', 'A'],
                                       right=False)  # right=False means left-closed intervals

    return df_quintiles