# client_corrected.py
import flwr as fl
import torch
import numpy as np
from collections import OrderedDict
from model import DiabetesNet
import pandas as pd
from sklearn.preprocessing import StandardScaler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLIENTS = 2
CLIENT_ID = int(input("Enter client ID (0 or 1): "))

# Initialize model
model = DiabetesNet().to(DEVICE)

# Load data directly (simplified)
def load_client_data(client_id, num_clients=2):
    df = pd.read_csv("diabetes.csv")
    
    # Clean data
    zero_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for feature in zero_features:
        if (df[feature] == 0).sum() > 0:
            median_val = df[df[feature] != 0][feature].median()
            df[feature] = df[feature].replace(0, median_val)
    
    # Features and labels
    X = df.drop("Outcome", axis=1).values
    y = df["Outcome"].values.reshape(-1, 1)
    
    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split (80% train, 20% test)
    split_idx = int(0.8 * len(X))
    
    # Split data between clients for training
    client_train_size = split_idx // num_clients
    start = client_id * client_train_size
    end = (client_id + 1) * client_train_size if client_id < num_clients - 1 else split_idx
    
    # Client gets its portion of training data
    X_train = X[start:end]
    y_train = y[start:end]
    
    # All clients use the same test set
    X_test = X[split_idx:]
    y_test = y[split_idx:]
    
    print(f"\nClient {client_id}:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Diabetic in training: {y_train.sum()}/{len(y_train)} ({y_train.sum()/len(y_train)*100:.1f}%)")
    print(f"  Diabetic in test: {y_test.sum()}/{len(y_test)} ({y_test.sum()/len(y_test)*100:.1f}%)")
    
    # Convert to DataLoader
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32)
    )
    
    return (torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True),
            torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False))

trainloader, testloader = load_client_data(CLIENT_ID, NUM_CLIENTS)

def get_parameters(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

def train(net, trainloader, epochs=5):
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    criterion = torch.nn.BCELoss()
    
    for epoch in range(epochs):
        total_loss = 0
        for X, y in trainloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(X).view(-1)
            loss = criterion(outputs, y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Calculate training accuracy
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in trainloader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                outputs = net(X).view(-1)
                preds = (outputs > 0.5).float()
                correct += (preds == y.view(-1)).sum().item()
                total += y.size(0)
        
        print(f"[Client {CLIENT_ID}] Epoch {epoch+1}/{epochs} - "
              f"Loss: {total_loss/len(trainloader):.4f}, "
              f"Train Acc: {correct/total:.4f}")

def evaluate_model(net, testloader, threshold=0.5):
    net.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    criterion = torch.nn.BCELoss()
    
    with torch.no_grad():
        for X, y in testloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            outputs = net(X).view(-1)
            
            # Calculate loss
            loss = criterion(outputs, y.view(-1))
            total_loss += loss.item()
            
            # Get predictions
            preds = (outputs > threshold).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy().flatten())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    correct = (all_preds == all_labels).sum()
    total = len(all_labels)
    accuracy = correct / total if total > 0 else 0
    
    # Calculate confusion matrix
    tp = ((all_preds == 1) & (all_labels == 1)).sum()
    fp = ((all_preds == 1) & (all_labels == 0)).sum()
    tn = ((all_preds == 0) & (all_labels == 0)).sum()
    fn = ((all_preds == 0) & (all_labels == 1)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    avg_loss = total_loss / len(testloader)
    
    print(f"\n[Client {CLIENT_ID}] Evaluation:")
    print(f"  Loss: {avg_loss:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1: {f1:.4f}")
    print(f"  TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
    print(f"  Predictions: {all_preds.sum()}/{total} diabetic "
          f"({all_preds.sum()/total*100:.1f}%)")
    print(f"  Actual: {all_labels.sum()}/{total} diabetic "
          f"({all_labels.sum()/total*100:.1f}%)")
    
    # Return in the CORRECT format: (float, int, dict)
    return float(avg_loss), int(total), {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }

class DiabetesClient(fl.client.NumPyClient):
    def __init__(self):
        super().__init__()
    
    def get_parameters(self, config):
        return get_parameters(model)
    
    def fit(self, parameters, config):
        set_parameters(model, parameters)
        train(model, trainloader, epochs=5)
        return get_parameters(model), len(trainloader.dataset), {}
    
    def evaluate(self, parameters, config):
        # CRITICAL: This must return (float, int, dict)
        set_parameters(model, parameters)
        
        # Get threshold from config or use default
        threshold = config.get("threshold", 0.5)
        
        # Call evaluate_model which returns the correct format
        loss, num_examples, metrics = evaluate_model(model, testloader, threshold)
        
        # Ensure we return the correct types
        return float(loss), int(num_examples), metrics

# Start the client
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=DiabetesClient())