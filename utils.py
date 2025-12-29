# utils_fixed.py
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def clean_diabetes_data(df):
    """Handle missing/zero values in diabetes dataset"""
    df_clean = df.copy()
    
    # Features where 0 is not biologically plausible
    # Replace zeros with median values
    zero_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    for feature in zero_features:
        # Only replace zeros if they exist
        if (df_clean[feature] == 0).sum() > 0:
            median_val = df_clean[df_clean[feature] != 0][feature].median()
            df_clean[feature] = df_clean[feature].replace(0, median_val)
            print(f"Replaced zeros in {feature} with median: {median_val:.2f}")
    
    return df_clean

def load_data(client_id, num_clients=2):
    # Read dataset
    df = pd.read_csv("diabetes.csv")
    
    # Clean the data
    df = clean_diabetes_data(df)
    
    print(f"\n=== CLIENT {client_id} ===")
    print(f"Total samples in dataset: {len(df)}")
    print(f"Class distribution - Non-diabetic: {(df['Outcome']==0).sum()} ({(df['Outcome']==0).sum()/len(df)*100:.1f}%)")
    print(f"Class distribution - Diabetic: {(df['Outcome']==1).sum()} ({(df['Outcome']==1).sum()/len(df)*100:.1f}%)")
    
    # Features and labels
    X = df.drop("Outcome", axis=1).values
    y = df["Outcome"].values.reshape(-1, 1)
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Train-test split (stratified to maintain class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nAfter train-test split:")
    print(f"Training set: {len(X_train)} samples")
    train_positive = (y_train == 1).sum()
    print(f"  Diabetic in training: {train_positive} ({train_positive/len(y_train)*100:.1f}%)")
    
    # Divide training data among clients
    chunk_size = len(X_train) // num_clients
    start = client_id * chunk_size
    end = (client_id + 1) * chunk_size if client_id < num_clients - 1 else len(X_train)
    
    # Each client gets a portion of the data
    X_client = X_train[start:end]
    y_client = y_train[start:end]
    
    print(f"\nClient {client_id} data:")
    print(f"  Samples: {len(X_client)}")
    client_positive = (y_client == 1).sum()
    print(f"  Diabetic: {client_positive} ({client_positive/len(y_client)*100:.1f}%)")
    
    # Handle severe class imbalance with weighted sampling
    if client_positive == 0 or client_positive == len(y_client):
        print(f"⚠️  WARNING: Client {client_id} has only one class!")
        # If only one class, don't use weighted sampler
        train_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_client, dtype=torch.float32),
                torch.tensor(y_client, dtype=torch.float32)
            ),
            batch_size=16,
            shuffle=True
        )
    else:
        # Calculate class weights for handling imbalance
        class_counts = np.bincount(y_client.flatten().astype(int))
        print(f"  Class counts: {class_counts}")
        
        # Higher weight for minority class
        class_weights = 1. / class_counts
        class_weights = class_weights / class_weights.sum() * len(class_counts)
        
        print(f"  Class weights: {class_weights}")
        
        sample_weights = class_weights[y_client.flatten().astype(int)]
        
        # Create sampler
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        train_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_client, dtype=torch.float32),
                torch.tensor(y_client, dtype=torch.float32)
            ),
            batch_size=16,
            sampler=sampler
        )
    
    # Test loader (no sampling, just evaluation)
    test_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32)
        ),
        batch_size=16,
        shuffle=False
    )
    
    return train_loader, test_loader, class_weights if 'class_weights' in locals() else None