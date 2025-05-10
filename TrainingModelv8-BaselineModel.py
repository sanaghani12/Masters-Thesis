import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score,  precision_score, recall_score, f1_score
from datetime import datetime
import pandas as pd

# Paths for data
before_grids_folder = 'D:\\sana\\Thesis\\Savir_Data\\output_gridsog\\before_grids'
after_grids_folder = 'D:\\sana\\Thesis\\Savir_Data\\output_gridsog\\after_grids'
before_responses_folder = 'D:\\sana\\Thesis\\Savir_Data\\output_gridsog\\before_responses'
after_responses_folder = 'D:\\sana\\Thesis\\Savir_Data\\output_gridsog\\after_responses'

# Grid size
grid_shape = (15, 19)

# Load data from .npy files
def load_npy_files(folder):
    data = []
    for file in sorted(os.listdir(folder)):
        if file.endswith('.npy'):
            file_path = os.path.join(folder, file)
            data.append(np.load(file_path))
    return np.array(data)

def normalize_responses_per_patient(responses):
    """
    Normalize response times for each patient separately while ensuring:
    - Zero values are replaced with (max + 10ms)
    - Normalization is done based on per-patient min (excluding 0) and max.

    Args:
        responses (numpy array): 3D array of response times (patients, grid_height, grid_width).

    Returns:
        tuple: (normalized responses, min values per patient, max values per patient)
    """
    processed_responses, min_vals, max_vals = [], [], []

    for response in responses:
        # Replace 0 with (max + 10ms)
        response[response == 0] = np.max(response) + 10
        # Find min and max per patient **excluding 0**
        min_val = np.min(response[response > 0]) if np.any(response > 0) else 0
        max_val = np.max(response)

        # Normalize using per-patient min and max
        if max_val > min_val:  # Prevent division by zero
            normalized_response = (response - min_val) / (max_val - min_val)
        else:
            normalized_response = response  # No normalization if min == max

        # Store min/max values for unnormalization later
        min_vals.append(min_val)
        max_vals.append(max_val)
        processed_responses.append(normalized_response)

    return np.array(processed_responses), np.array(min_vals), np.array(max_vals)

# Data preprocessing
def prepare_dataset():
    before_grids = load_npy_files(before_grids_folder)
    after_grids = load_npy_files(after_grids_folder)
    before_responses = load_npy_files(before_responses_folder)
    after_responses = load_npy_files(after_responses_folder)
    
    # Normalize data
    before_grids = before_grids / np.max(before_grids)
    after_grids = after_grids / np.max(after_grids)
    '''before_responses = (before_responses - np.min(before_responses)) / (np.max(before_responses) - np.min(before_responses))
    after_responses = (after_responses - np.min(after_responses)) / (np.max(after_responses) - np.min(after_responses))'''
    before_response_normalized, before_min_vals, before_max_vals = normalize_responses_per_patient(before_responses)
    after_response_normalized, after_min_vals, after_max_vals = normalize_responses_per_patient(after_responses)


    # Stack before grids and responses into two-channel input
    inputs = np.stack([before_grids, before_response_normalized], axis=-1)

    # Train-test split (splitting min/max values alongside data)
    '''X_train, X_test, y_train_grids, y_test_grids, y_train_responses, y_test_responses, before_min_test, before_max_test = train_test_split(
        inputs, after_grids, after_response_normalized, before_min_vals, before_max_vals, test_size=0.2, random_state=42
    )'''
      # First split: Main data
    X_train, X_test, y_train_grids, y_test_grids, y_train_responses, y_test_responses = train_test_split(
        inputs, after_grids, after_response_normalized, test_size=0.2, random_state=42
    )

    # Second split: Min/Max values (matching the same split as above)
    before_min_train, before_min_test, before_max_train, before_max_test = train_test_split(
        before_min_vals, before_max_vals, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train_grids, y_test_grids, y_train_responses, y_test_responses, before_min_test, before_max_test

def dice_score(y_true, y_pred, smooth=1e-6):
    """
    Compute the Dice Score for multi-class classification.

    Args:
        y_true (numpy array): True labels (integer values).
        y_pred (numpy array): Predicted labels (integer values).
        smooth (float): Small value to avoid division by zero.

    Returns:
        float: Dice coefficient.
    """
    y_true_f = np.array(y_true).flatten()
    y_pred_f = np.array(y_pred).flatten()

    intersection = np.sum(y_true_f == y_pred_f)
    dice = (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

    return dice

def jaccard_index(y_true, y_pred):
    intersection = np.sum((y_true.flatten() == y_pred.flatten()) & (y_true.flatten() != 0))
    union = np.sum((y_true.flatten() != 0) | (y_pred.flatten() != 0))
    return intersection / union if union != 0 else 0

def discretize_grid(grid):
    """ Convert continuous grid values into discrete classes """
    discretized = np.zeros_like(grid)
    discretized[grid == 0.0] = 0  # Black
    discretized[grid == 0.5] = 1  # Grey
    discretized[grid == 1.0] = 2  # White
    return discretized

def baseline_model(X_test, y_test_grids, y_test_responses, y_train_grids, y_train_responses, before_min_test, before_max_test, method):
    """
    Enhanced baseline model that outputs metrics in the same format as U-Net for direct comparison.
    """
    if method not in ["persistence"]:
        raise ValueError("Invalid method.")

    # Extract before-treatment values (persistence baseline)
    before_grids = X_test[..., 0]
    before_responses = X_test[..., 1]

    # Make predictions (using persistence method)
    y_pred_grids = before_grids
    y_pred_responses = before_responses

    # Unnormalize responses for proper metric calculation
    unnormalized_true_responses = []
    unnormalized_pred_responses = []
    for i in range(len(X_test)):
        min_val, max_val = before_min_test[i], before_max_test[i]
        true_response = (y_test_responses[i] * (max_val - min_val)) + min_val
        pred_response = (y_pred_responses[i] * (max_val - min_val)) + min_val
        unnormalized_true_responses.append(true_response)
        unnormalized_pred_responses.append(pred_response)

    unnormalized_true_responses = np.array(unnormalized_true_responses)
    unnormalized_pred_responses = np.array(unnormalized_pred_responses)

    # Convert to categorical classes for grid metrics
    y_test_labels = discretize_grid(y_test_grids)
    y_pred_labels = discretize_grid(y_pred_grids)

    # Calculate per-class metrics
    per_class_metrics = {}
    for class_idx, class_name in enumerate(['Black', 'Grey', 'White']):
        class_mask_true = (y_test_labels == class_idx)
        class_mask_pred = (y_pred_labels == class_idx)
        
        tp = np.sum(class_mask_true & class_mask_pred)
        fp = np.sum(~class_mask_true & class_mask_pred)
        fn = np.sum(class_mask_true & ~class_mask_pred)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        per_class_metrics[class_name] = {
            'Precision': precision,
            'Recall': recall,
            'F1-score': f1
        }

    # Calculate grid metrics
    grid_metrics = {
        'Accuracy': accuracy_score(y_test_labels.flatten(), y_pred_labels.flatten()),
        'F1 Score': f1_score(y_test_labels.flatten(), y_pred_labels.flatten(), average='weighted'),
        'Dice Score': dice_score(y_test_labels.flatten(), y_pred_labels.flatten()),
        'Jaccard Index': jaccard_index(y_test_labels, y_pred_labels),
        'Per-class': per_class_metrics
    }

    # Calculate response metrics
    response_metrics = {
        'MAE': mean_absolute_error(unnormalized_true_responses.flatten(), unnormalized_pred_responses.flatten()),
        'RMSE': np.sqrt(mean_squared_error(unnormalized_true_responses.flatten(), unnormalized_pred_responses.flatten())),
        # 'R² Score': r2_score(unnormalized_true_responses.flatten(), unnormalized_pred_responses.flatten()),
        'No-Response Accuracy': accuracy_score(
            (unnormalized_true_responses.flatten() == 0),
            (unnormalized_pred_responses.flatten() == 0)
        )
    }

    return {
        'original_grids': grid_metrics,
        'responses': response_metrics
    }

def save_baseline_metrics_to_csv(baseline_results, filename=None):
    """
    Save baseline model metrics to a CSV file.
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"baseline_metrics_{timestamp}.csv"
    
    # Create metrics dictionary
    metrics_dict = {
        'Model': 'Baseline_Persistence',
        'Timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        
        # Grid Classification Metrics
        'Accuracy': baseline_results['original_grids']['Accuracy'],
        'Dice_Score': baseline_results['original_grids']['Dice Score'],
        'Jaccard_Index': baseline_results['original_grids']['Jaccard Index'],
        
        # Per-class Metrics
        'Black_Precision': baseline_results['original_grids']['Per-class']['Black']['Precision'],
        'Black_Recall': baseline_results['original_grids']['Per-class']['Black']['Recall'],
        'Black_F1': baseline_results['original_grids']['Per-class']['Black']['F1-score'],
        
        'Grey_Precision': baseline_results['original_grids']['Per-class']['Grey']['Precision'],
        'Grey_Recall': baseline_results['original_grids']['Per-class']['Grey']['Recall'],
        'Grey_F1': baseline_results['original_grids']['Per-class']['Grey']['F1-score'],
        
        'White_Precision': baseline_results['original_grids']['Per-class']['White']['Precision'],
        'White_Recall': baseline_results['original_grids']['Per-class']['White']['Recall'],
        'White_F1': baseline_results['original_grids']['Per-class']['White']['F1-score'],
        
        # Response Time Metrics
        'MAE': baseline_results['responses']['MAE'],
        'RMSE': baseline_results['responses']['RMSE'],
        'No_Response_Accuracy': baseline_results['responses']['No-Response Accuracy']
    }
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame([metrics_dict])
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    csv_path = os.path.join('results', filename)
    
    # If file exists, append to it, otherwise create new
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)
    
    print(f"\nMetrics saved to: {csv_path}")
    return csv_path

# Modify the end of your script to save the metrics
if __name__ == "__main__":
    # Run the pipeline
    X_train, X_test, y_train_grids, y_test_grids, y_train_responses, y_test_responses, before_min_test, before_max_test = prepare_dataset()
    baseline_results = baseline_model(X_test, y_test_grids, y_test_responses, y_train_grids, y_train_responses, 
                                    before_min_test, before_max_test, method="persistence")
    
    # Save metrics to CSV
    csv_path = save_baseline_metrics_to_csv(baseline_results)
    
    # Print results in a formatted way
    print("\nBASELINE MODEL RESULTS")
    print("=" * 50)

    print("\nGrid Classification Metrics:")
    print("-" * 40)
    print(f"Accuracy: {baseline_results['original_grids']['Accuracy']:.4f}")
    print(f"F1 Score: {baseline_results['original_grids']['F1 Score']:.4f}")
    print(f"Dice Score: {baseline_results['original_grids']['Dice Score']:.4f}")
    print(f"Jaccard Index: {baseline_results['original_grids']['Jaccard Index']:.4f}")

    print("\nPer-class Metrics:")
    print("  {:<10} {:<12} {:<12} {:<12}".format("Class", "Precision", "Recall", "F1-score"))
    print("  " + "-" * 46)
    for class_name, metrics in baseline_results['original_grids']['Per-class'].items():
        print("  {:<10} {:<12.4%} {:<12.4%} {:<12.4%}".format(
            class_name,
            metrics['Precision'],
            metrics['Recall'],
            metrics['F1-score']
        ))

    print("\nResponse Time Metrics:")
    print("-" * 40)
    print(f"MAE: {baseline_results['responses']['MAE']:.4f}")
    print(f"RMSE: {baseline_results['responses']['RMSE']:.4f}")
    # print(f"R² Score: {baseline_results['responses']['R² Score']:.4f}")
    print(f"No-Response Accuracy: {baseline_results['responses']['No-Response Accuracy']:.4f}")
