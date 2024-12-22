# Data
import ast
import pandas as pd
import numpy as np
from typing import Dict
# ML
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF
# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def explore_qualitative_variables(df, variables):
    for var in variables:
        if var not in df.columns:
            print(f"Variable '{var}' not found in DataFrame columns.")
            continue
            
        # Calculate frequency count
        freq_count = df[var].value_counts()
        
        # Calculate participation rate
        participation = freq_count / len(df) * 100

        # Prepare DataFrame for plotting
        summary_df = pd.DataFrame({
            'Frequency': freq_count,
            'Participation (%)': participation
        }).reset_index().rename(columns={'index': var})

        # Plotting
        fig, ax1 = plt.subplots(figsize=(10, 5))

        # Plot frequency as bars on the left y-axis
        sns.barplot(x=var, y='Frequency', data=summary_df, ax=ax1, color='blue', alpha=0.6, label='Frequency')
        ax1.set_ylabel('Frequency', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        # Create a second y-axis for participation
        ax2 = ax1.twinx()
        sns.lineplot(x=var, y='Participation (%)', data=summary_df, ax=ax2, color='orange', marker='o', label='Participation (%)')
        ax2.set_ylabel('Participation (%)', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')

        plt.title(f'Frequency and Participation of {var}')
        plt.xlabel(var)

        # Set correct x-ticks and x-tick labels
        ax1.set_xticks(range(len(summary_df[var])))  # Ensure tick positions correspond to categories
        ax1.set_xticklabels(summary_df[var], rotation=45)  # Set the tick labels

        plt.show()
        
        
def explore_quantitive_variables(data, variables):
    
    df = data[variables]
    
    # Do a summary statistics
    summary_statistics = df.describe()
    print(summary_statistics)

    # Create a pairplot for scatter plots and histograms
    sns.pairplot(df)
    plt.suptitle('Pair Plot of Values', y=1.02)  # Title above the plot
    plt.show()

    # Estimate Correlations
    correlation_matrix = df.corr()

    # Plot the heatmap of the correlation coefficient matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, square=True)
    plt.title('Correlation Coefficient Matrix')
    plt.show()


def check_unique_combinations(df, fields):
    if not fields:
        raise ValueError("The list of fields cannot be empty.")
    unique_counts = df.groupby(fields).size()
    is_unique = all(unique_counts == 1)
    return is_unique


def find_duplicate_ids(df, fields):
    if not fields:
        raise ValueError("The list of fields cannot be empty.")
    duplicates = df[df.duplicated(subset=fields, keep=False)]
    return duplicates