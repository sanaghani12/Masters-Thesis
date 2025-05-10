import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg before importing pyplot
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score, confusion_matrix,mean_absolute_error
from sklearn.metrics import precision_score, recall_score, jaccard_score
from scipy.ndimage import uniform_filter, generic_filter
import seaborn as sns
import os
import re
from datetime import datetime
import pandas as pd

def load_npy_files(folder):
    data = []
    filenames = []
    for file in sorted(os.listdir(folder)):
        if file.endswith('.npy'):
            file_path = os.path.join(folder, file)
            data.append(np.load(file_path))
            filenames.append(file)
    return np.array(data), filenames

def extract_patient_details(filename):
    pattern = re.compile(r"combined_grid_(?P<name>.+?)_(?P<id>NV\d+_\d+)_(?P<eye>Right Eye|Left Eye)_date_(?P<date>\d{4}-\d{2}-\d{2})")
    match = pattern.search(filename)
    if match:
        full_name = match.group("name").replace("_", " ")
        patient_id = match.group("id")
        eye = match.group("eye")
        date = match.group("date")
        return full_name, patient_id, eye, date
    return "Unknown", "Unknown", "Unknown", "Unknown"

def local_std(arr):
    return np.std(arr)

def prepare_features(before_grid, before_response):
    height, width = before_grid.shape
    n_pixels = height * width
    
    # Basic features
    rows = np.arange(height).repeat(width)
    cols = np.tile(np.arange(width), height)
    norm_rows = rows / height
    norm_cols = cols / width
    
    # Create response mask (1 for valid response, 0 for no response)
    response_mask = (before_response > 0).astype(float)
    
    # Grid-level statistics
    grid_mean = np.mean(before_grid)
    grid_std = np.std(before_grid)
    
    # Response-level statistics (only using valid responses)
    valid_responses = before_response[before_response > 0]
    response_mean = np.mean(valid_responses) if len(valid_responses) > 0 else 0
    response_std = np.std(valid_responses) if len(valid_responses) > 0 else 0
    
    # Replace zero responses with mean for feature calculation
    masked_response = np.where(before_response > 0, before_response, response_mean)

    # Pad grid to safely extract 3x3 patches at edges
    padded_grid = np.pad(before_grid, pad_width=1, mode='reflect')

    # Compute 3x3 mean (neighborhood activity)
    neighborhood_activity = uniform_filter(padded_grid, size=3)[1:-1, 1:-1]

    # Compute 3x3 std (homogeneity)
    # homogeneity = generic_filter(padded_grid, local_std, size=3)[1:-1, 1:-1]

    # Combine features
    features = np.column_stack([
        before_grid.flatten(),                    # Current pixel value
        masked_response.flatten(),                # Masked response time
        response_mask.flatten(),                  # Response mask (0/1)
        norm_rows, norm_cols,                     # Normalized position
        np.repeat(grid_mean, n_pixels),          # Grid statistics
        np.repeat(grid_std, n_pixels),
        np.repeat(response_mean, n_pixels),      # Response statistics
        np.repeat(response_std, n_pixels),
        neighborhood_activity.flatten(), 
        # homogeneity.flatten()                
    ])
    
    return features

def train_random_forest_models(X_train, y_train_grid, y_train_response):
    # Reshape targets if needed
    y_train_grid = y_train_grid.reshape(-1)
    y_train_response = y_train_response.reshape(-1)
    
    # For regression, only use valid responses
    response_mask = y_train_response > 0
    X_train_response = X_train[response_mask]
    y_train_response = y_train_response[response_mask]

    # Calculate class distribution
    unique, counts = np.unique(y_train_grid, return_counts=True)
    class_dist = dict(zip(unique, counts))
    total_samples = len(y_train_grid)
    
    print("\nClass distribution in training data:")
    print(f"Black (0): {class_dist[0]} ({class_dist[0]/total_samples:.2%})")
    print(f"Grey (1): {class_dist[1]} ({class_dist[1]/total_samples:.2%})")
    print(f"White (2): {class_dist[2]} ({class_dist[2]/total_samples:.2%})")
    
    # Define class weights with higher weight for grey class
    class_weights = {
        0: 1.0,    # Black
        1: 2.6,    # Grey (higher weight)
        2: 1.3     # White
    }
    print("\nUsing class weights:", class_weights)
    
    # Train classifier with class weights
    rf_classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=30,
        random_state=42,
        n_jobs=-1,
        class_weight=class_weights  # Add class weights
    )
    
    # Train regressor only on valid responses
    rf_regressor = RandomForestRegressor(
        n_estimators=100,
        max_depth=30,
        random_state=42,
        n_jobs=-1
    )
    
    print("\nTraining Random Forest Classifier...")
    rf_classifier.fit(X_train, y_train_grid)
    
    print("\nTraining Random Forest Regressor...")
    print(f"Training on {len(y_train_response)} valid response times")
    rf_regressor.fit(X_train_response, y_train_response)
    
    return rf_classifier, rf_regressor

def compute_dice(y_true, y_pred, num_classes=3):
    dice_scores = []
    for c in range(num_classes):
        y_true_c = (y_true == c).astype(int)
        y_pred_c = (y_pred == c).astype(int)
        intersection = np.sum(y_true_c * y_pred_c)
        denom = np.sum(y_true_c) + np.sum(y_pred_c)
        dice = (2. * intersection) / denom if denom != 0 else 1.0
        dice_scores.append(dice)
    return np.mean(dice_scores)

def compute_jaccard(y_true, y_pred):
    return jaccard_score(y_true, y_pred, average='weighted', zero_division=1)

def evaluate_models(classifier, regressor, X_test, y_test_grid, y_test_response):
    grid_shape = (15, 19)
    pixels_per_sample = 15 * 19
    
    # Make predictions
    grid_pred = classifier.predict(X_test)
    
    # Calculate Dice scores for each class
    '''def dice_score(y_true, y_pred, class_id):
        y_true_class = (y_true == class_id)
        y_pred_class = (y_pred == class_id)
        intersection = np.sum(y_true_class & y_pred_class)
        total = np.sum(y_true_class) + np.sum(y_pred_class)
        if total == 0:
            return 1.0
        return 2.0 * intersection / total
    
    dice_scores = {
        'black': dice_score(y_test_grid, grid_pred, 0),
        'grey': dice_score(y_test_grid, grid_pred, 1),
        'white': dice_score(y_test_grid, grid_pred, 2),
    }'''
    
    # For regression, only evaluate on pixels with actual responses
    response_mask = y_test_response > 0
    X_test_response = X_test[response_mask]
    y_test_response_valid = y_test_response[response_mask]
    
    print(f"\nResponse rate in test data: {np.mean(response_mask):.2%}")
    print(f"Number of valid responses: {np.sum(response_mask)}")
    print(f"Number of no responses: {np.sum(~response_mask)}")
    
    # Make response predictions for all pixels
    response_pred_all = np.zeros(len(X_test))  # Initialize with zeros for no-response pixels
    response_pred_valid = regressor.predict(X_test_response)
    response_pred_all[response_mask] = response_pred_valid
    
    # Overall metrics
    accuracy = accuracy_score(y_test_grid, grid_pred)
    f1 = f1_score(y_test_grid, grid_pred, average='weighted')
    dice = compute_dice(y_test_grid, grid_pred)
    jaccard = compute_jaccard(y_test_grid, grid_pred)

    # Per-class metrics
    precision_per_class = precision_score(y_test_grid, grid_pred, average=None, labels=[0, 1, 2], zero_division=0)
    recall_per_class = recall_score(y_test_grid, grid_pred, average=None, labels=[0, 1, 2], zero_division=0)
    f1_per_class = f1_score(y_test_grid, grid_pred, average=None, labels=[0, 1, 2], zero_division=0)



    # Calculate metrics
    metrics = {
        'per_class': {
        'precision': precision_per_class.tolist(),
        'recall': recall_per_class.tolist(),
        'f1_score': f1_per_class.tolist()
        },
        'classification': {
            'accuracy': accuracy,
            'f1_score': f1,
            'dice_scores': dice,
            'jaccard_scores': jaccard
        },
        'regression': {
            'mse': mean_squared_error(y_test_response_valid, response_pred_valid),
            'mae': mean_absolute_error(y_test_response_valid, response_pred_valid),
            'rmse': np.sqrt(mean_squared_error(y_test_response_valid, response_pred_valid)),
            'r2_score': r2_score(y_test_response_valid, response_pred_valid),
            'response_rate': np.mean(response_mask)
        }
    }
    
    # Print detailed classification metrics
    print("\nClassification Metrics:")
    print(f"Accuracy:      {accuracy:.4f}")
    print(f"F1 Score:      {f1:.4f}")
    print(f"Dice Score:    {dice:.4f}")
    print(f"Jaccard Index: {jaccard:.4f}")

    print("\nPer-Class Metrics:")
    print("  Class      Precision    Recall       F1-score")
    print("  ----------------------------------------------")
    classes = ['Black', 'Grey', 'White']
    for i, cls in enumerate(classes):
        print(f"  {cls:<10}{precision_per_class[i]*100:>7.2f}%     {recall_per_class[i]*100:>7.2f}%     {f1_per_class[i]*100:>7.2f}%")
    
    # Print regression metrics
    print("\nRegression Metrics:")
    print(f"MSE: {metrics['regression']['mse']:.4f}")
    print(f"RMSE: {metrics['regression']['rmse']:.4f}")
    print(f"R² Score: {metrics['regression']['r2_score']:.4f}")
    print(f"Response Rate: {metrics['regression']['response_rate']:.2%}")
    
    # Reshape predictions back to grid format for visualization
    n_samples = len(X_test) // pixels_per_sample
    grid_pred = grid_pred.reshape(n_samples, 15, 19)
    response_pred = response_pred_all.reshape(n_samples, 15, 19)
    
    return metrics, grid_pred, response_pred

def visualize_results(before_grid, true_grid, pred_grid, true_response, pred_response, 
                     patient_details, save_path):
    fig, axs = plt.subplots(2, 3, figsize=(20, 12))
    
    # Plot input grid
    axs[0,0].imshow(before_grid, cmap='gray')
    axs[0,0].set_title('Before Treatment Grid')
    
    # Plot true after grid
    axs[0,1].imshow(true_grid, cmap='gray')
    axs[0,1].set_title('True After Grid')
    
    # Plot predicted grid
    axs[0,2].imshow(pred_grid, cmap='gray')
    axs[0,2].set_title('Predicted After Grid')
    
    # Plot true response times
    axs[1,0].imshow(true_response, cmap='coolwarm')
    axs[1,0].set_title('True Response Times')
    
    # Plot predicted response times
    axs[1,1].imshow(pred_response, cmap='coolwarm')
    axs[1,1].set_title('Predicted Response Times')
    
    # Add error map
    error = np.abs(true_response - pred_response)
    im = axs[1,2].imshow(error, cmap='Reds')
    axs[1,2].set_title('Response Time Error')
    plt.colorbar(im, ax=axs[1,2])
    
    # Add patient details
    patient_name, patient_id, eye, date = patient_details
    plt.suptitle(f"Patient: {patient_name}\nID: {patient_id} | Eye: {eye} | Date: {date}", y=1.02)
    
    # Save and close
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def cross_validate_models(before_grids, after_grids, before_responses, after_responses, n_splits=5):
    """
    Perform k-fold cross-validation on both  classification and regression models.
    Returns mean and std of performance metrics across folds.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Lists to store metrics for each fold
    fold_metrics = []
    confusion_matrices = []
    
    print(f"\nPerforming {n_splits}-fold cross-validation...")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(before_grids)):
        print(f"\nFold {fold + 1}/{n_splits}")
        
        # Split data
        X_train_fold = []
        y_train_grid_fold = []
        y_train_response_fold = []
        X_val_fold = []
        y_val_grid_fold = []
        y_val_response_fold = []
        
        # Process training data
        for i in train_idx:
            features = prepare_features(before_grids[i], before_responses[i])
            X_train_fold.append(features)
            y_train_grid_fold.append(after_grids[i].flatten())
            y_train_response_fold.append(after_responses[i].flatten())
        
        # Process validation data
        for i in val_idx:
            features = prepare_features(before_grids[i], before_responses[i])
            X_val_fold.append(features)
            y_val_grid_fold.append(after_grids[i].flatten())
            y_val_response_fold.append(after_responses[i].flatten())
        
        # Stack arrays
        X_train = np.vstack(X_train_fold)
        y_train_grid = np.concatenate(y_train_grid_fold)
        y_train_response = np.concatenate(y_train_response_fold)
        
        X_val = np.vstack(X_val_fold)
        y_val_grid = np.concatenate(y_val_grid_fold)
        y_val_response = np.concatenate(y_val_response_fold)
        
        # Train models
        classifier, regressor = train_random_forest_models(
            X_train, y_train_grid, y_train_response
        )
        
        # Evaluate models
        metrics, _, _ = evaluate_models(
            classifier, regressor, X_val, y_val_grid, y_val_response
        )
        fold_metrics.append(metrics)
        
        # Calculate confusion matrix for this fold
        y_pred = classifier.predict(X_val)
        cm = confusion_matrix(y_val_grid, y_pred)
        confusion_matrices.append(cm)
    
    # Calculate mean and std of metrics across folds
    mean_metrics = {
        'classification': {
            'accuracy': np.mean([m['classification']['accuracy'] for m in fold_metrics]),
            'accuracy_std': np.std([m['classification']['accuracy'] for m in fold_metrics]),
            'f1_score': np.mean([m['classification']['f1_score'] for m in fold_metrics]),
            'f1_score_std': np.std([m['classification']['f1_score'] for m in fold_metrics]),
            'dice_score': np.mean([m['classification']['dice_scores'] for m in fold_metrics]),
            'dice_score_std': np.std([m['classification']['dice_scores'] for m in fold_metrics])
        },
        'regression': {
            'mse': np.mean([m['regression']['mse'] for m in fold_metrics]),
            'mse_std': np.std([m['regression']['mse'] for m in fold_metrics]),
            'rmse': np.mean([m['regression']['rmse'] for m in fold_metrics]),
            'rmse_std': np.std([m['regression']['rmse'] for m in fold_metrics]),
            'mae': np.mean([m['regression']['mae'] for m in fold_metrics]),
            'mae_std': np.std([m['regression']['mae'] for m in fold_metrics]),
            'r2_score': np.mean([m['regression']['r2_score'] for m in fold_metrics]),
            'r2_score_std': np.std([m['regression']['r2_score'] for m in fold_metrics])
        }
    }
    
     # Aggregate per-class metrics
    class_labels = ['Black', 'Grey', 'White']
    per_class_metrics = {'precision': [], 'recall': [], 'f1_score': []}

    for fold in fold_metrics:
        for metric in ['precision', 'recall', 'f1_score']:
            per_class_metrics[metric].append(fold['per_class'][metric])

    # Convert to numpy arrays for mean/std computation
    per_class_metrics = {k: np.array(v) for k, v in per_class_metrics.items()}  # shape: (folds, 3)

    # Compute mean and std per class
    per_class_summary = {metric: {} for metric in ['precision', 'recall', 'f1_score']}
    for metric in ['precision', 'recall', 'f1_score']:
        for i, cls in enumerate(class_labels):
            per_class_summary[metric][cls] = {
                'mean': float(np.mean(per_class_metrics[metric][:, i])),
                'std': float(np.std(per_class_metrics[metric][:, i]))
            }

    # Include in mean_metrics
    mean_metrics['per_class'] = per_class_summary

    # Calculate mean confusion matrix
    mean_cm = np.mean(confusion_matrices, axis=0)
    
    return mean_metrics, mean_cm

def visualize_confusion_matrix(confusion_matrix, save_path):
    """
    Visualize confusion matrix with class labels.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt='.0f', cmap='Blues',
                xticklabels=['Black', 'Grey', 'White'],
                yticklabels=['Black', 'Grey', 'White'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def main():
    # Paths for data
    before_grids_folder = 'D:\\sana\\Thesis\\Savir_Data\\output_gridsog\\before_grids'
    after_grids_folder = 'D:\\sana\\Thesis\\Savir_Data\\output_gridsog\\after_grids'
    before_responses_folder = 'D:\\sana\\Thesis\\Savir_Data\\output_gridsog\\before_responses'
    after_responses_folder = 'D:\\sana\\Thesis\\Savir_Data\\output_gridsog\\after_responses'
    
    # Load data
    before_grids, before_grid_files = load_npy_files(before_grids_folder)
    after_grids, _ = load_npy_files(after_grids_folder)
    before_responses, _ = load_npy_files(before_responses_folder)
    after_responses, _ = load_npy_files(after_responses_folder)
    
    print("Data shapes:")
    print(f"Before grids: {before_grids.shape}")
    print(f"After grids: {after_grids.shape}")
    print(f"Before responses: {before_responses.shape}")
    print(f"After responses: {after_responses.shape}")
    
    # Convert grids to categorical labels (0, 1, 2)
    def to_labels(grid):
        labels = np.zeros_like(grid, dtype=np.int64)
        labels[grid == 0.0] = 0   # Black
        labels[grid == 0.5] = 1   # Grey
        labels[grid == 1.0] = 2   # White
        return labels
    
    before_grids = to_labels(before_grids)
    after_grids = to_labels(after_grids)
    
    # Split patients into train and test sets first
    n_patients = len(before_grids)
    patient_indices = np.arange(n_patients)
    train_patient_indices, test_patient_indices = train_test_split(
        patient_indices, test_size=0.2, random_state=42
    )
    
    # Prepare features and targets separately for train and test
    train_features = []
    train_grid_targets = []
    train_response_targets = []
    test_features = []
    test_grid_targets = []
    test_response_targets = []
    
    # Process training patients
    for i in train_patient_indices:
        features = prepare_features(before_grids[i], before_responses[i])
        train_features.append(features)
        train_grid_targets.append(after_grids[i].flatten())
        train_response_targets.append(after_responses[i].flatten())
    
    # Process test patients
    for i in test_patient_indices:
        features = prepare_features(before_grids[i], before_responses[i])
        test_features.append(features)
        test_grid_targets.append(after_grids[i].flatten())
        test_response_targets.append(after_responses[i].flatten())
    
    # Stack all samples
    X_train = np.vstack(train_features)
    y_train_grid = np.concatenate(train_grid_targets)
    y_train_response = np.concatenate(train_response_targets)
    
    X_test = np.vstack(test_features)
    y_test_grid = np.concatenate(test_grid_targets)
    y_test_response = np.concatenate(test_response_targets)
    
    print("\nProcessed data shapes:")
    print(f"Training samples: {len(train_patient_indices)} patients")
    print(f"Test samples: {len(test_patient_indices)} patients")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    
    # Perform cross-validation first
    cv_metrics, mean_cm = cross_validate_models(
        before_grids, after_grids, before_responses, after_responses, n_splits=5
    )
    
    # Print cross-validation results
    print("\nCross-validation Results (mean ± std):")
    print("\nClassification Metrics:")
    print(f"Accuracy: {cv_metrics['classification']['accuracy']:.4f} ± {cv_metrics['classification']['accuracy_std']:.4f}")
    print(f"F1 Score: {cv_metrics['classification']['f1_score']:.4f} ± {cv_metrics['classification']['f1_score_std']:.4f}")
    print(f"Dice Scores: {cv_metrics['classification']['dice_score']:.4f} ± {cv_metrics['classification']['dice_score_std']:.4f}")
    
    
    '''print("\nPer-Class Metrics:")
    for cls in ['Black', 'Grey', 'White']:
        print(f"\nClass: {cls}")
        print(f"Precision: {cv_metrics['per_class']['precision'][cls]['mean']:.4f} ± {cv_metrics['per_class']['precision'][cls]['std']:.4f}")
        print(f"Recall: {cv_metrics['per_class']['recall'][cls]['mean']:.4f} ± {cv_metrics['per_class']['recall'][cls]['std']:.4f}")
        print(f"F1 Score: {cv_metrics['per_class']['f1_score'][cls]['mean']:.4f} ± {cv_metrics['per_class']['f1_score'][cls]['std']:.4f}")'''
    

    print("\nPer-Class Metrics:")
    print("  Class      Precision    Recall       F1-score")
    print("  ----------------------------------------------")

    for cls in ['Black', 'Grey', 'White']:
        precision = cv_metrics['per_class']['precision'][cls]['mean'] * 100
        recall = cv_metrics['per_class']['recall'][cls]['mean'] * 100
        f1 = cv_metrics['per_class']['f1_score'][cls]['mean'] * 100
        print(f"  {cls:<10}{precision:>10.2f}%   {recall:>10.2f}%   {f1:>10.2f}%")


    print("\nRegression Metrics:")
    print(f"MSE: {cv_metrics['regression']['mse']:.4f} ± {cv_metrics['regression']['mse_std']:.4f}")
    print(f"MAE: {cv_metrics['regression']['mae']:.4f} ± {cv_metrics['regression']['mae_std']:.4f}")
    print(f"RMSE: {cv_metrics['regression']['rmse']:.4f} ± {cv_metrics['regression']['rmse_std']:.4f}")
    print(f"R² Score: {cv_metrics['regression']['r2_score']:.4f} ± {cv_metrics['regression']['r2_score_std']:.4f}")
    
    # Save confusion matrix visualization
    os.makedirs('rf_predictions', exist_ok=True)
    visualize_confusion_matrix(mean_cm, 'rf_predictions/confusion_matrix.png')
    
    # Continue with the final model training and evaluation
    classifier, regressor = train_random_forest_models(
        X_train, y_train_grid, y_train_response
    )
    
    # Evaluate models
    metrics, grid_predictions, response_predictions = evaluate_models(
        classifier, regressor, X_test, y_test_grid, y_test_response
    )
    
    # Save predictions
    os.makedirs('rf_predictions', exist_ok=True)
    
    # Visualize results for test patients
    for i, test_idx in enumerate(test_patient_indices):
        patient_details = extract_patient_details(before_grid_files[test_idx])
        save_path = f"rf_predictions/pred_{patient_details[1]}_{patient_details[2].replace(' ', '_')}.png"
        
        visualize_results(
            before_grids[test_idx],
            after_grids[test_idx],
            grid_predictions[i],
            after_responses[test_idx],
            response_predictions[i],
            patient_details,
            save_path
        )
    
    # Plot feature importance
    feature_names = [
        'Pixel Value',          # Current pixel value
        'Response Time',        # Masked response time
        'Response Mask',        # Response mask (0/1)
        'Row Position',         # Normalized row position
        'Col Position',         # Normalized column position
        'Grid Mean',           # Grid statistics
        'Grid Std',
        'Response Mean',       # Response statistics
        'Response Std',
        'Neighborhood Activity'
        # 'Neighborhood Homogeneity'         
    ]
    
    plt.figure(figsize=(20, 8))
    
    # Plot classifier feature importance
    plt.subplot(1, 2, 1)
    plt.title('Grid Classification Feature Importance')
    importance_clf = pd.Series(classifier.feature_importances_, index=feature_names)
    importance_clf.sort_values().plot(kind='barh')
    plt.xlabel('Importance')
    
    # Plot regressor feature importance
    plt.subplot(1, 2, 2)
    plt.title('Response Time Feature Importance')
    importance_reg = pd.Series(regressor.feature_importances_, index=feature_names)
    importance_reg.sort_values().plot(kind='barh')
    plt.xlabel('Importance')
    
    plt.tight_layout()
    plt.savefig('rf_feature_importance.png', bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    main() 