import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class UserDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class CollaborativeFilteringModel(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64, 32]):
        super(CollaborativeFilteringModel, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Create hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        # Output layer for offer group predictions
        self.output_layer = nn.Linear(hidden_dims[-1], 10)  # 10 offer groups
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.model(x)
        return torch.sigmoid(self.output_layer(x))

def prepare_data(df):
    """
    Prepare the data for training the model
    """
    # Select feature columns
    feature_cols = [
        'customer_type', 'amount', 'offer_completed', 
        'offer_received', 'offer_viewed', 'transaction', 'time'
    ]
    
    # Target columns (offer groups)
    target_cols = [f'offer_group_{i}' for i in range(10)]
    
    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_cols])
    y = df[target_cols].values
    
    return X, y, scaler

def train_model(model, train_loader, val_loader, epochs=50, learning_rate=0.001):
    """
    Train the collaborative filtering model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                outputs = model(features)
                val_loss += criterion(outputs, targets).item()
        
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

def get_recommendations(model, user_features, scaler, top_n=5):
    """
    Get offer group recommendations for a user
    """
    model.eval()
    with torch.no_grad():
        # Scale the features
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
    X, y, scaler = prepare_data(df)
    
    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create data loaders
    train_dataset = UserDataset(X_train, y_train)
    val_dataset = UserDataset(X_val, y_val)
    
    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False
    )
    
    # Initialize and train the model
    model = CollaborativeFilteringModel(input_dim=7)  # 7 input features
    trained_model = train_model(model, train_loader, val_loader)
    
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
    
    recommendations = get_recommendations(
        trained_model, sample_user_features, scaler
    )
    return recommendations