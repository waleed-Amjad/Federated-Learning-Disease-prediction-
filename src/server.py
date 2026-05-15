# server_simple.py
import flwr as fl
import matplotlib.pyplot as plt
import pickle
import torch
from collections import OrderedDict
from model import DiabetesNet
import numpy as np

# Store metrics
round_accuracies = []
round_losses = []

def weighted_average(metrics):
    """Aggregate metrics from clients"""
    # Extract metrics
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    losses = [num_examples * m.get("loss", 0) for num_examples, m in metrics]
    total_examples = sum(num_examples for num_examples, _ in metrics)
    
    # Calculate weighted averages
    avg_accuracy = sum(accuracies) / total_examples if total_examples > 0 else 0
    avg_loss = sum(losses) / total_examples if total_examples > 0 else 0
    
    # Store for plotting
    round_accuracies.append(avg_accuracy)
    round_losses.append(avg_loss)
    
    print(f"\nRound {len(round_accuracies)}:")
    print(f"  Average Accuracy: {avg_accuracy:.4f}")
    print(f"  Average Loss: {avg_loss:.4f}")
    
    # Return aggregated metrics
    return {
        "accuracy": avg_accuracy,
        "loss": avg_loss
    }

class SimpleFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_model_params = None
    
    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        if aggregated_parameters is not None:
            aggregated_ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)
            self.global_model_params = aggregated_ndarrays
            
            # Save model after last round
            if server_round == 10:
                self.save_model(aggregated_ndarrays)
        
        return aggregated_parameters, aggregated_metrics
    
    def save_model(self, parameters):
        """Save model to files"""
        # Save as pickle
        with open('global_model.pkl', 'wb') as f:
            pickle.dump(parameters, f)
        
        # Save as PyTorch model
        model = DiabetesNet()
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        
        torch.save(model.state_dict(), 'global_model.pth')
        print(f"\n✅ Model saved successfully!")
        print(f"   - global_model.pkl (Flower format)")
        print(f"   - global_model.pth (PyTorch format)")

# Define strategy
strategy = SimpleFedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2,
    evaluate_metrics_aggregation_fn=weighted_average,
)

print("Starting Federated Learning Server...")
print("=" * 50)

# Start server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy
)

# Plot results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, len(round_accuracies) + 1), round_accuracies, 
         marker='o', color='blue', linewidth=2)
plt.title("Accuracy over Rounds")
plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.grid(True, alpha=0.3)
plt.ylim([0, 1])

plt.subplot(1, 2, 2)
plt.plot(range(1, len(round_losses) + 1), round_losses,
         marker='s', color='red', linewidth=2)
plt.title("Loss over Rounds")
plt.xlabel("Round")
plt.ylabel("Loss")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_results.png', dpi=100, bbox_inches='tight')
plt.show()

# Print final results
print("\n" + "=" * 50)
print("FINAL RESULTS")
print("=" * 50)
if round_accuracies:
    print(f"Final Accuracy: {round_accuracies[-1]:.4f}")
    print(f"Final Loss: {round_losses[-1]:.4f}")
    print(f"Total Rounds: {len(round_accuracies)}")