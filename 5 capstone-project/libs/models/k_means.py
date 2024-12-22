import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def generate_model_kmeans_clusters(df, n_clusters=3, random_state=42, plot=True):    
    # Select relevant features
    categorical_features = ['customer_type']
    numerical_features = list(set(df.columns) - set(categorical_features) - set(["profile_id"]))
    
    # Create a copy of the input dataframe
    X = df.copy()

    # One-hot encode the customer_type column
    encoder = OneHotEncoder(sparse_output=False)
    customer_type_encoded = encoder.fit_transform(X[categorical_features])

    # Create a DataFrame with correct column names
    customer_type_encoded_df = pd.DataFrame(customer_type_encoded, columns=encoder.get_feature_names_out(['customer_type']))

    # Combine numerical features and encoded customer types
    X_combined = pd.concat([X[numerical_features], customer_type_encoded_df], axis=1)

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
    df_with_clusters['cluster'] = cluster_labels

    if plot:
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