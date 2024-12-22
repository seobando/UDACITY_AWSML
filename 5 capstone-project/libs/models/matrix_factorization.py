import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class MatrixFactorizationDataset(Dataset):
    def __init__(self, user_features, offer_interactions):
        self.user_features = torch.FloatTensor(user_features)
        self.offer_interactions = torch.FloatTensor(offer_interactions)
    
    def __len__(self):
        return len(self.user_features)
    
    def __getitem__(self, idx):
        return self.user_features[idx], self.offer_interactions[idx]

class MatrixFactorizationModel(nn.Module):
    def __init__(self, n_users, n_offers, n_factors=50, user_features_dim=7):
        super(MatrixFactorizationModel, self).__init__()
        
        # User embedding layers
        self.user_features_layer = nn.Sequential(
            nn.Linear(user_features_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_factors)
        )
        
        # Offer embedding layer
        self.offer_embeddings = nn.Parameter(
            torch.randn(n_offers, n_factors) * 0.01
        )
        
        # Bias terms
        self.user_biases = nn.Parameter(torch.zeros(n_users))
        self.offer_biases = nn.Parameter(torch.zeros(n_offers))
        self.global_bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, user_features):
        # Get user embeddings from features
        user_factors = self.user_features_layer(user_features)
        
        # Calculate predictions
        predictions = torch.mm(user_factors, self.offer_embeddings.t())
        
        # Add biases
        predictions += self.offer_biases.unsqueeze(0)
        predictions += self.user_biases.unsqueeze(1)
        predictions += self.global_bias
        
        return torch.sigmoid(predictions)

def prepare_data(df):
    """
    Prepare user features and offer interactions matrix
    """
    # Prepare user features
    feature_cols = [
        'customer_type', 'amount', 'offer_completed', 
        'offer_received', 'offer_viewed', 'transaction', 'time'
    ]
    
    # Scale features
    scaler = StandardScaler()
    user_features = scaler.fit_transform(df[feature_cols])
    
    # Create offer interactions matrix
    offer_cols = [f'offer_group_{i}' for i in range(10)]
    offer_interactions = df[offer_cols].values
    
    return user_features, offer_interactions, scaler

def train_matrix_factorization(model, train_loader, val_loader, epochs=50, learning_rate=0.001):
    """
    Train the matrix factorization model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for user_features, interactions in train_loader:
            user_features = user_features.to(device)
            interactions = interactions.to(device)
            
            optimizer.zero_grad()
            predictions = model(user_features)
            loss = criterion(predictions, interactions)
            
            # Add L2 regularization for embeddings
            l2_reg = torch.norm(model.offer_embeddings) + torch.norm(model.user_biases) + torch.norm(model.offer_biases)
            loss += 0.01 * l2_reg
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for user_features, interactions in val_loader:
                user_features = user_features.to(device)
                interactions = interactions.to(device)
                predictions = model(user_features)
                val_loss += criterion(predictions, interactions).item()
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
            
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Train Loss = {train_loss/len(train_loader):.4f}, Val Loss = {val_loss/len(val_loader):.4f}")
    
    return model

def get_recommendations_mf(model, user_features, scaler, top_n=5):
    """
    Get offer recommendations using the matrix factorization model
    """
    model.eval()
    with torch.no_grad():
        # Scale user features
        scaled_features = scaler.transform(user_features.reshape(1, -1))
        features = torch.FloatTensor(scaled_features)
        
        # Get predictions
        predictions = model(features).numpy()[0]
        
        # Get top N recommendations
        top_indices = np.argsort(predictions)[-top_n:][::-1]
        top_scores = predictions[top_indices]
        
        recommendations = [
            (f"offer_group_{idx}", score) 
            for idx, score in zip(top_indices, top_scores)
        ]
    
    return recommendations

def main():
    # Load your data
    # df = pd.read_csv('your_data.csv')
    
    # Prepare the data
    user_features, offer_interactions, scaler = prepare_data(df)
    
    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(
        user_features, offer_interactions, test_size=0.2, random_state=42
    )
    
    # Create data loaders
    train_dataset = MatrixFactorizationDataset(X_train, y_train)
    val_dataset = MatrixFactorizationDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize and train the model
    n_users = len(user_features)
    n_offers = 10  # number of offer groups
    model = MatrixFactorizationModel(n_users, n_offers, n_factors=50)
    
    trained_model = train_matrix_factorization(
        model, train_loader, val_loader
    )
    
    # Example of getting recommendations for a user
    sample_user_features = np.array([
        1,      # customer_type
        100.0,  # amount
        5,      # offer_completed
        10,     # offer_received
        8,      # offer_viewed
        15,     # transaction
        48.0    # time
    ])
    
    recommendations = get_recommendations_mf(
        trained_model, sample_user_features, scaler
    )
    return recommendations