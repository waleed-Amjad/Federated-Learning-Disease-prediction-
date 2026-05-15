# analyze_data.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Create output directory
output_dir = "analysis_outputs"
os.makedirs(output_dir, exist_ok=True)

# Load data
df = pd.read_csv("diabetes.csv")

print("=== Dataset Info ===")
print(f"Shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nData types:\n{df.dtypes}")

print("\n=== Class Distribution ===")
class_counts = df['Outcome'].value_counts()
print(class_counts)
print(f"\nPercentage:")
print(f"Non-diabetic: {class_counts[0]/len(df)*100:.1f}%")
print(f"Diabetic: {class_counts[1]/len(df)*100:.1f}%")

print("\n=== Basic Statistics ===")
print(df.describe())

print("\n=== Missing Values ===")
print(df.isnull().sum())

print("\n=== Correlation with Outcome ===")
correlations = df.corr()['Outcome'].sort_values(ascending=False)
print(correlations)

# Plot correlations
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlations')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'feature_correlations.png'))

# Plot class distribution
plt.figure(figsize=(8, 5))
sns.countplot(x='Outcome', data=df)
plt.title('Class Distribution (0=Non-diabetic, 1=Diabetic)')
plt.xlabel('Outcome')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'class_distribution.png'))

# Plot feature distributions by class
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
features = df.columns[:-1]

for idx, feature in enumerate(features):
    ax = axes[idx//3, idx%3]
    df.boxplot(column=feature, by='Outcome', ax=ax)
    ax.set_title(f'{feature} by Outcome')
    ax.set_xlabel('Outcome')
    ax.set_ylabel(feature)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'feature_distributions.png'))
plt.show()

print("\n=== Recommendations ===")
print("1. Check for class imbalance (common in medical datasets)")
print("2. Consider feature engineering based on correlations")
print("3. Try different model architectures")
print("4. Use appropriate evaluation metrics (precision, recall, F1)")
print("5. Consider data augmentation or resampling techniques")
