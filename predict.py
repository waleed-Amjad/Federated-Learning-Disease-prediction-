# predict.py
import torch
import pickle
import numpy as np
import pandas as pd
import argparse
from collections import OrderedDict
from model import DiabetesNet
from sklearn.preprocessing import StandardScaler

def load_model(model_path='global_model.pth'):
    """Load the trained global model"""
    # Initialize model
    model = DiabetesNet()
    
    # Load the saved state dict
    if model_path.endswith('.pth'):
        # Load PyTorch model
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print(f"✓ Model loaded from {model_path}")
    elif model_path.endswith('.pkl'):
        # Load from pickle file (Flower format)
        with open(model_path, 'rb') as f:
            parameters = pickle.load(f)
        
        # Create state dictionary
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        
        # Load state dict into model
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        print(f"✓ Model loaded from {model_path}")
    else:
        raise ValueError(f"Unsupported file format: {model_path}")
    
    return model

def clean_diabetes_data(df):
    """Clean the diabetes data - handle zero values"""
    df_clean = df.copy()
    
    # Replace zeros with median values for medical features where zero is not plausible
    zero_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    for feature in zero_features:
        if feature in df_clean.columns:
            if (df_clean[feature] == 0).sum() > 0:
                median_val = df_clean[df_clean[feature] != 0][feature].median()
                df_clean[feature] = df_clean[feature].replace(0, median_val)
                print(f"  - Replaced zeros in {feature} with median: {median_val:.2f}")
    
    return df_clean

def prepare_features(df, expected_features=None):
    """Prepare features for prediction"""
    if expected_features is None:
        # Default PIMA diabetes features
        expected_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    
    # Check if we have the expected features
    missing_features = set(expected_features) - set(df.columns)
    if missing_features:
        print(f"⚠️  Warning: Missing features: {missing_features}")
        print(f"   Available features: {list(df.columns)}")
        
        # Try to use first N columns as features
        if len(df.columns) >= 8:
            print(f"   Using first 8 columns as features")
            feature_columns = df.columns[:8].tolist()
        else:
            raise ValueError(f"Need at least 8 feature columns, got {len(df.columns)}")
    else:
        feature_columns = expected_features
    
    # Extract features
    features = df[feature_columns].values
    
    return features, feature_columns

def predict_single(model, features, threshold=0.5):
    """Make prediction for a single sample"""
    model.eval()
    with torch.no_grad():
        # Convert features to tensor
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        
        # Make prediction
        output = model(features_tensor)
        probability = output.item()
        prediction = 1 if probability > threshold else 0
        
    return {
        'prediction': int(prediction),
        'probability': float(probability),
        'class': 'Diabetic' if prediction == 1 else 'Non-Diabetic'
    }

def predict_batch(model, features_list, threshold=0.5):
    """Make predictions for multiple samples"""
    model.eval()
    with torch.no_grad():
        # Convert to tensor
        features_tensor = torch.tensor(features_list, dtype=torch.float32)
        
        # Make predictions
        outputs = model(features_tensor).squeeze()
        probabilities = outputs.numpy()
        predictions = (probabilities > threshold).astype(int)
    
    results = []
    for i in range(len(features_list)):
        results.append({
            'sample': i+1,
            'prediction': int(predictions[i]),
            'probability': float(probabilities[i]),
            'class': 'Diabetic' if predictions[i] == 1 else 'Non-Diabetic'
        })
    
    return results

def predict_from_csv(model, csv_path, threshold=0.5, output_suffix='_predictions', 
                    clean_data=True, scale_features=True):
    """
    Load data from CSV and make predictions
    Saves results in a CSV file similar to your original format
    """
    print(f"\n📂 Loading data from: {csv_path}")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"  - Found {len(df)} samples with {len(df.columns)} columns")
    print(f"  - Columns: {list(df.columns)}")
    
    # Clean data if requested
    if clean_data:
        df_clean = clean_diabetes_data(df)
        print(f"  - Data cleaned (zero values handled)")
    else:
        df_clean = df.copy()
    
    # Prepare features
    features, feature_columns = prepare_features(df_clean)
    print(f"  - Using features: {feature_columns}")
    
    # Scale features if requested (recommended)
    if scale_features:
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        print(f"  - Features standardized (StandardScaler)")
    else:
        features_scaled = features
    
    # Make predictions
    print(f"  - Making predictions with threshold: {threshold}")
    results = predict_batch(model, features_scaled, threshold)
    
    # Add predictions to dataframe
    df_result = df.copy()  # Keep original data
    df_result['prediction'] = [r['prediction'] for r in results]
    df_result['probability'] = [r['probability'] for r in results]
    df_result['class'] = [r['class'] for r in results]
    
    # Calculate statistics
    diabetic_count = df_result['prediction'].sum()
    non_diabetic_count = len(df_result) - diabetic_count
    
    print(f"\n📊 Prediction Statistics:")
    print(f"  - Total samples: {len(df_result)}")
    print(f"  - Predicted Diabetic: {diabetic_count} ({diabetic_count/len(df_result)*100:.1f}%)")
    print(f"  - Predicted Non-Diabetic: {non_diabetic_count} ({non_diabetic_count/len(df_result)*100:.1f}%)")
    print(f"  - Average probability: {df_result['probability'].mean():.3f}")
    print(f"  - Probability range: [{df_result['probability'].min():.3f}, {df_result['probability'].max():.3f}]")
    
    # Save results
    if csv_path.endswith('.csv'):
        output_path = csv_path.replace('.csv', f'{output_suffix}.csv')
    else:
        output_path = f"{csv_path}{output_suffix}.csv"
    
    df_result.to_csv(output_path, index=False)
    print(f"\n💾 Predictions saved to {output_path}")
    
    return df_result

def predict_from_csv_with_analysis(model, csv_path, threshold=0.5, has_labels=False):
    """
    Advanced version: Load data from CSV, make predictions, and analyze results
    """
    print(f"\n🔍 Advanced CSV Analysis")
    print("=" * 60)
    
    # Load and predict
    df_result = predict_from_csv(model, csv_path, threshold, output_suffix='_predictions_advanced')
    
    # If we have true labels, calculate metrics
    if has_labels and 'Outcome' in df_result.columns:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        
        true_labels = df_result['Outcome'].values
        pred_labels = df_result['prediction'].values
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, pred_labels)
        precision = precision_score(true_labels, pred_labels, zero_division=0)
        recall = recall_score(true_labels, pred_labels, zero_division=0)
        f1 = f1_score(true_labels, pred_labels, zero_division=0)
        cm = confusion_matrix(true_labels, pred_labels)
        
        print(f"\n📈 Evaluation Metrics (Threshold: {threshold}):")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        
        print(f"\n📊 Confusion Matrix:")
        print(f"                Predicted")
        print(f"               0     1")
        print(f"Actual   0   {cm[0,0]:4d}  {cm[0,1]:4d}")
        print(f"         1   {cm[1,0]:4d}  {cm[1,1]:4d}")
        
        # Analyze threshold sensitivity
        print(f"\n🎯 Threshold Sensitivity Analysis:")
        for t in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
            preds_t = (df_result['probability'].values > t).astype(int)
            acc_t = accuracy_score(true_labels, preds_t)
            diabetic_pct = preds_t.sum() / len(preds_t) * 100
            print(f"  Threshold {t:.2f}: Accuracy={acc_t:.4f}, Diabetic={diabetic_pct:.1f}%")
    
    return df_result

def batch_predict_multiple_csvs(model, csv_files, threshold=0.5):
    """Make predictions for multiple CSV files"""
    results_summary = []
    
    for csv_file in csv_files:
        print(f"\n{'='*60}")
        print(f"Processing: {csv_file}")
        print('='*60)
        
        try:
            df_result = predict_from_csv(model, csv_file, threshold)
            
            # Summarize results
            summary = {
                'file': csv_file,
                'total_samples': len(df_result),
                'diabetic_count': df_result['prediction'].sum(),
                'diabetic_percent': df_result['prediction'].sum() / len(df_result) * 100,
                'avg_probability': df_result['probability'].mean()
            }
            results_summary.append(summary)
            
        except Exception as e:
            print(f"❌ Error processing {csv_file}: {e}")
            results_summary.append({
                'file': csv_file,
                'error': str(e)
            })
    
    # Print summary
    if results_summary:
        print(f"\n{'='*60}")
        print("BATCH PROCESSING SUMMARY")
        print('='*60)
        
        summary_df = pd.DataFrame(results_summary)
        print(summary_df.to_string(index=False))

def main():
    parser = argparse.ArgumentParser(description='Diabetes Prediction Tool')
    parser.add_argument('--model', type=str, default='global_model.pth',
                       help='Path to trained model (default: global_model.pth)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Prediction threshold (default: 0.5)')
    parser.add_argument('--csv', type=str, nargs='+',
                       help='Path to CSV file(s) for batch prediction')
    parser.add_argument('--single', nargs=8, type=float, metavar=('Preg', 'Glu', 'BP', 'Skin', 
                       'Insulin', 'BMI', 'DPF', 'Age'),
                       help='Make single prediction with 8 feature values')
    parser.add_argument('--batch', nargs='+', type=float, 
                       help='Make batch prediction with multiple feature sets')
    parser.add_argument('--has-labels', action='store_true',
                       help='CSV file has "Outcome" column with true labels')
    parser.add_argument('--no-clean', action='store_true',
                       help='Skip data cleaning (zero value handling)')
    parser.add_argument('--no-scale', action='store_true',
                       help='Skip feature scaling')
    
    args = parser.parse_args()
    
    # Load the trained model
    model = load_model(args.model)
    
    # Single prediction mode
    if args.single:
        print("\n=== Single Prediction ===")
        print(f"Threshold: {args.threshold}")
        print(f"Features: {args.single}")
        
        result = predict_single(model, args.single, args.threshold)
        print(f"\nResult:")
        print(f"  Prediction: {result['class']}")
        print(f"  Probability: {result['probability']:.4f}")
        print(f"  Confidence: {'HIGH' if result['probability'] > 0.7 or result['probability'] < 0.3 else 'MEDIUM' if result['probability'] > 0.6 or result['probability'] < 0.4 else 'LOW'}")
    
    # Batch prediction from command line
    elif args.batch:
        print("\n=== Batch Prediction ===")
        # Reshape batch features
        batch_features = np.array(args.batch).reshape(-1, 8)
        print(f"Making predictions for {len(batch_features)} samples")
        print(f"Threshold: {args.threshold}")
        
        results = predict_batch(model, batch_features, args.threshold)
        for r in results:
            print(f"Sample {r['sample']}: {r}")
    
    # CSV prediction mode
    elif args.csv:
        if len(args.csv) == 1:
            # Single CSV file
            if args.has_labels:
                df = predict_from_csv_with_analysis(model, args.csv[0], args.threshold, has_labels=True)
            else:
                df = predict_from_csv(model, args.csv[0], args.threshold, 
                                     clean_data=not args.no_clean, 
                                     scale_features=not args.no_scale)
        else:
            # Multiple CSV files
            batch_predict_multiple_csvs(model, args.csv, args.threshold)
    
    # Interactive examples (default behavior)
    else:
        print("\n=== Diabetes Prediction Examples ===")
        
        # Load model
        print(f"Model loaded: {args.model}")
        print(f"Default threshold: {args.threshold}")
        
        # Example 1: Single prediction
        print("\n1. Single Prediction Example:")
        example_features = [6, 148, 72, 35, 0, 33.6, 0.627, 50]
        result = predict_single(model, example_features, args.threshold)
        print(f"   Features: {example_features}")
        print(f"   Result: {result}")
        
        # Example 2: Batch prediction
        print("\n2. Batch Prediction Example:")
        batch_features = [
            [6, 148, 72, 35, 0, 33.6, 0.627, 50],
            [1, 85, 66, 29, 0, 26.6, 0.351, 31],
            [8, 183, 64, 0, 0, 23.3, 0.672, 32]
        ]
        results = predict_batch(model, batch_features, args.threshold)
        for r in results:
            print(f"   Sample {r['sample']}: {r}")
        
        # Example 3: CSV prediction instructions
        print("\n3. CSV Prediction Instructions:")
        print("   To predict from a CSV file, run:")
        print("   python predict.py --csv your_data.csv --threshold 0.35")
        print("\n   If your CSV has true labels:")
        print("   python predict.py --csv your_data.csv --has-labels --threshold 0.35")
        print("\n   For multiple CSV files:")
        print("   python predict.py --csv data1.csv data2.csv data3.csv")

if __name__ == "__main__":
    main()