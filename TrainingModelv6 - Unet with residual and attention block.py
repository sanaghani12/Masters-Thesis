import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils import shuffle
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, jaccard_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout, BatchNormalization, Conv2DTranspose, Resizing, Cropping2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import cv2
import re
from tensorflow.keras import backend as K
import pandas as pd
from datetime import datetime
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from sklearn.metrics import confusion_matrix
import seaborn as sns


# Paths for data
before_grids_folder = 'D:\\sana\\Thesis\\Savir_Data\\output_gridsog\\before_grids'
after_grids_folder = 'D:\\sana\\Thesis\\Savir_Data\\output_gridsog\\after_grids'
before_responses_folder = 'D:\\sana\\Thesis\\Savir_Data\\output_gridsog\\before_responses'
after_responses_folder = 'D:\\sana\\Thesis\\Savir_Data\\output_gridsog\\after_responses'
model_base_dir = "model_checkpoints-with-resunetandattention"
model_path = os.path.join(model_base_dir, "v6-trained_unet_mode_regression.keras")

# Grid size
grid_shape = (15, 19)

# Custom metric for masked accuracy
@tf.keras.utils.register_keras_serializable()
class Accuracy(tf.keras.metrics.Metric):
    def __init__(self, name='masked_accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        # Initialize state variables
        self.correct = self.add_weight(name='correct', initializer='zeros')
        self.total = self.add_weight(name='total', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None): 
        y_pred = tf.argmax(y_pred, axis=-1)  # Convert softmax output to class labels
        y_pred = tf.cast(y_pred, tf.int64)   # Ensure int64 for comparison       
        # Update counts
        self.correct.assign_add(
            tf.reduce_sum(tf.cast(tf.equal(y_true, y_pred), tf.float32))
        )
        self.total.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self):
        return self.correct / self.total  # Return accuracy

    def reset_state(self):
        # Reset counts between epochs
        self.correct.assign(0.0)
        self.total.assign(0.0)

# Define class frequencies from dataset

class_counts = {
    "black": 43132,
    "grey": 18199,
    "white": 68914
}

total_pixels = sum(class_counts.values())

# Function to compute α values using inverse frequency
def compute_alpha(class_counts, power):
    """ Compute alpha values using inverse frequency weighting. """
    total_pixels = sum(class_counts.values())
    return [((total_pixels / class_counts[key]) ** power) for key in class_counts.keys()]
        
@tf.keras.utils.register_keras_serializable()
def focal_loss(alpha, gamma=2.0):  
    if alpha is None:
        print("Computing α values using DEFAULT inverse frequency weighting...")
        alpha = compute_alpha(class_counts, power=1.0)  # Default α computation

    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, 1e-7, 1.0 - 1e-7)
        y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=len(alpha))

        alpha_factor = tf.constant(alpha, dtype=tf.float32)
        gamma_factor = tf.constant(gamma, dtype=tf.float32)

        focal_weight = alpha_factor * K.pow((1 - y_pred), gamma_factor)
        focal_loss = -focal_weight * y_true_one_hot * K.log(y_pred)

        return K.mean(K.sum(focal_loss, axis=-1))

    return loss

# Custom loss function
@tf.keras.utils.register_keras_serializable()
def sparse_categorical_crossentropy(y_true, y_pred):
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    return tf.reduce_mean(loss)

# Load data from .npy files and extract filenames
def load_npy_files(folder):
    data = []
    filenames = []
    for file in sorted(os.listdir(folder)):
        if file.endswith('.npy'):
            file_path = os.path.join(folder, file)
            data.append(np.load(file_path))
            filenames.append(file)
    return np.array(data), filenames
# Pad input tensor to (16, 20)
def pad_input(input_tensor):
    padded_tensor = tf.pad(
        input_tensor,
        paddings=[[0, 1], [0, 1], [0, 0]],  # Pad 1 row and 1 column
        mode='SYMMETRIC'  # Reflect padding to avoid artifacts
    )
    return padded_tensor
    return np.array(data), filenames

# Extract patient details from filename
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

def normalize_responses_per_patient(responses):
    """
    Normalize response times for each patient separately while ensuring:
    - Zero values (no response) are marked with a special value (2.0)
    - Normalization is done based on per-patient min and max of non-zero values
    - Preserves the distinction between no-response and actual responses

    Args:
        responses (numpy array): 3D array of response times (patients, grid_height, grid_width).

    Returns:
        tuple: (normalized responses, min values per patient, max values per patient)
    """
    processed_responses, min_vals, max_vals = [], [], []

    for response in responses:
        # Create mask for zero values
        zero_mask = (response == 0)
        
        # Get max and min of non-zero values
        max_val = np.max(response[~zero_mask]) if np.any(~zero_mask) else 0
        min_val = np.min(response[~zero_mask]) if np.any(~zero_mask) else 0
        
        # Normalize non-zero values
        normalized_response = np.copy(response)
        if max_val > min_val:
            normalized_response[~zero_mask] = (response[~zero_mask] - min_val) / (max_val - min_val)
        
        # Set zero values to a special value (2.0)
        normalized_response[zero_mask] = 2.0
        
        processed_responses.append(normalized_response)
        min_vals.append(min_val)
        max_vals.append(max_val)

    return np.array(processed_responses), min_vals, max_vals

def unnormalize_responses(normalized_responses, min_vals, max_vals):
    """
    Unnormalize response times while properly handling no-response values.

    Args:
        normalized_responses (numpy array): Normalized response times
        min_vals (array): Minimum values per patient
        max_vals (array): Maximum values per patient

    Returns:
        numpy array: Unnormalized response times with proper no-response values
    """
    unnormalized_responses = np.zeros_like(normalized_responses)
    
    for i, response in enumerate(normalized_responses):
        # Create mask for special no-response value
        no_response_mask = (response >= 2.0)
        
        # Unnormalize regular values
        unnormalized_responses[i] = response * (max_vals[i] - min_vals[i]) + min_vals[i]
        
        # Set no-response values back to 0
        unnormalized_responses[i][no_response_mask] = 0

    return unnormalized_responses

def update_after_responses_per_patient(responses):
    """
    Normalize response times for each patient separately while ensuring:
    - Zero values are replaced with (max + 10ms)
    - Normalization is done based on per-patient min (excluding 0) and max.

    Args:
        responses (numpy array): 3D array of response times (patients, grid_height, grid_width).

    Returns:
        numpy array: Normalized response times per patient.
    """
    processed_responses = []

    for response in responses:
        
        # Replace 0 with (max + 10ms)
        response[response == 0] = np.max(response) + 10
        processed_responses.append(response)

    return np.array(processed_responses)

# Data preparation
def prepare_dataset():
    # Load data
    before_grids, before_grid_files = load_npy_files(before_grids_folder)
    after_grids, after_grid_files = load_npy_files(after_grids_folder)
    before_responses, before_response_files = load_npy_files(before_responses_folder)
    after_responses, after_response_files = load_npy_files(after_responses_folder)

    # Convert grids to categorical labels
    def to_labels(grid):
        labels = np.zeros_like(grid, dtype=np.int64)
        labels[grid == 0.0] = 0   # Black
        labels[grid == 0.5] = 1   # Grey
        labels[grid == 1.0] = 2   # White
        return labels.astype(np.int64)
    
    # Convert to classification labels
    after_grids_labels = np.array(to_labels(after_grids))
    before_grids_labels = np.array(to_labels(before_grids))
    print("before_grids_labels: ",np.unique(before_grids_labels))

    # Normalize response times while ignoring artificial values
    '''before_response = (before_responses - np.min(before_responses)) / (np.max(before_responses) - np.min(before_responses))
    before_responses[before_responses == 0] = 1'''
     # Normalize response times separately for each patient
    before_response_normalized, before_min_vals, before_max_vals = normalize_responses_per_patient(before_responses)
    after_response_normalized, after_min_vals, after_max_vals = normalize_responses_per_patient(after_responses)
    # after_responses = update_after_responses_per_patient(after_responses) 


    # Replace zero values in after_responses (but don't normalize it)
    '''max_response = np.max(after_responses[after_responses < np.max(after_responses)])
    after_responses[after_responses == 0] = max_response + 1000  # Ensure consistency with before_responses'''

    # Prepare inputs
    inputs = np.stack([before_grids_labels, before_response_normalized], axis=-1)
    padded_inputs = np.array([pad_input(x) for x in inputs], dtype=np.float64)

    # Debug dtype
    print("Data Type Check Before Training:")
    print(f"after_grids_labels dtype: {after_grids_labels.dtype}") 
    print(f"before_grids_labels dtype: {before_grids_labels.dtype}")  
    print(f"padded_inputs dtype: {padded_inputs.dtype}")  

    before_targets = {'before_grids': before_grids_labels, 'before_responses': before_response_normalized}
    after_targets = {'after_grids': after_grids_labels, 'after_responses': after_response_normalized}

    # Store patient details along with min/max values
    patient_details = [
        {
            'metadata': extract_patient_details(filename),
            'before_min': before_min_vals[i],
            'before_max': before_max_vals[i],
            'after_min': after_min_vals[i],
            'after_max': after_max_vals[i]
        }
        for i, filename in enumerate(before_grid_files)
    ]

    return padded_inputs, before_targets, after_targets, patient_details, before_grid_files

# Residual block
def residual_block(input, num_filters):
    """
    Create a residual block with two convolutional layers and a skip connection.
    """
    shortcut = Conv2D(num_filters, (1, 1), padding='same')(input)
    shortcut = BatchNormalization()(shortcut)
    
    conv = Conv2D(num_filters, (3, 3), activation='relu', padding='same')(input)
    conv = BatchNormalization()(conv)
    conv = Conv2D(num_filters, (3, 3), activation='relu', padding='same')(conv)
    conv = BatchNormalization()(conv)
    
    output = tf.keras.layers.Add()([shortcut, conv])
    return tf.keras.layers.Activation('relu')(output)

# Attention mechanism
def attention_block(input, skip_connection, num_filters):
    """
    Create an attention gate to focus on relevant features from skip connection.
    """
    # Process input pathway
    g1 = Conv2D(num_filters, kernel_size=1)(input)  # gate
    g1 = BatchNormalization()(g1)
    
    # Process skip connection
    x1 = Conv2D(num_filters, kernel_size=1)(skip_connection)  # skip connection
    x1 = BatchNormalization()(x1)
    
    # Combine for attention
    psi = tf.keras.layers.Activation('relu')(tf.keras.layers.Add()([g1, x1]))
    psi = Conv2D(1, kernel_size=1)(psi)
    psi = BatchNormalization()(psi)
    psi = tf.keras.layers.Activation('sigmoid')(psi)
    
    # Apply attention to skip connection
    return tf.keras.layers.Multiply()([skip_connection, psi])

# Enhanced convolutional block with residual connection
def conv_block(input, num_filters):
    """
    Enhanced convolution block with residual connection.
    """
    return residual_block(input, num_filters)

# Enhanced encoder block with dimension control
def encoder_block(input, num_filters):
    """
    Enhanced encoder block with proper padding in max pooling.
    """
    conv = conv_block(input, num_filters)
    pool = MaxPooling2D((2, 2), padding='same')(conv)  # Add padding='same' to maintain dimensions
    return pool, conv

# Enhanced decoder block with attention and dimension control
def decoder_block(input, skip_features, num_filters):
    """
    Enhanced decoder block with proper dimension handling.
    """
    # Calculate target dimensions
    target_height = input.shape[1] * 2
    target_width = input.shape[2] * 2
    
    # Upsample with transposed convolution
    upsampled = Conv2DTranspose(
        num_filters, 
        kernel_size=(2, 2), 
        strides=(2, 2), 
        padding='same'
    )(input)
    
    # Resize skip connection if dimensions don't match
    if skip_features.shape[1:3] != (target_height, target_width):
        skip_features = Resizing(
            target_height,
            target_width,
            interpolation='bilinear'
        )(skip_features)
    
    # Add attention mechanism
    attended_skip_features = attention_block(upsampled, skip_features, num_filters)
    
    # Concatenate and apply convolutions
    merged = concatenate([upsampled, attended_skip_features], axis=-1)
    x = conv_block(merged, num_filters)
    
    return x

# Enhanced U-Net architecture
def build_unet():
    inputs = Input(shape=(16, 20, 2))
    print(f"Input shape: {inputs.shape}")
    
    # Encoder path with dimension tracking and proper padding
    # Level 1: 16x20 -> 8x10
    p1, s1 = encoder_block(inputs, 64)
    print(f"Encoder Level 1 - Skip connection shape: {s1.shape}, Output shape: {p1.shape}")
    
    # Level 2: 8x10 -> 4x5
    p2, s2 = encoder_block(p1, 128)
    print(f"Encoder Level 2 - Skip connection shape: {s2.shape}, Output shape: {p2.shape}")
    
    # Level 3: 4x5 -> 2x3
    p3, s3 = encoder_block(p2, 256)
    print(f"Encoder Level 3 - Skip connection shape: {s3.shape}, Output shape: {p3.shape}")
    
    # Level 4: 2x3 -> 1x2
    p4, s4 = encoder_block(p3, 512)
    print(f"Encoder Level 4 - Skip connection shape: {s4.shape}, Output shape: {p4.shape}")
    
    # Bottleneck
    bn = conv_block(p4, 1024)
    print(f"Bottleneck shape: {bn.shape}")
    
    # Decoder path with careful upsampling
    # Level 4: 1x2 -> 2x3
    d1 = decoder_block(bn, s4, 512)
    print(f"Decoder Level 4 shape: {d1.shape}")
    
    # Level 3: 2x3 -> 4x5
    d2 = decoder_block(d1, s3, 256)
    print(f"Decoder Level 3 shape: {d2.shape}")
    
    # Level 2: 4x5 -> 8x10
    d3 = decoder_block(d2, s2, 128)
    print(f"Decoder Level 2 shape: {d3.shape}")
    
    # Level 1: 8x10 -> 16x20
    d4 = decoder_block(d3, s1, 64)
    print(f"Decoder Level 1 shape: {d4.shape}")
    
    # Process features for each output head
    grid_features = Conv2D(32, (3, 3), padding='same', activation='relu')(d4)
    grid_features = BatchNormalization()(grid_features)
    grid_features = Conv2D(32, (3, 3), padding='same', activation='relu')(grid_features)
    print(f"Grid features shape: {grid_features.shape}")
    
    response_features = Conv2D(32, (3, 3), padding='same', activation='relu')(d4)
    response_features = BatchNormalization()(response_features)
    response_features = Conv2D(32, (3, 3), padding='same', activation='relu')(response_features)
    print(f"Response features shape: {response_features.shape}")
    
    # Final outputs with correct cropping to 15x19
    grid_output = Conv2D(3, (1, 1), activation='softmax')(grid_features)
    response_output = Conv2D(1, (1, 1), activation='linear')(response_features)
    
    # Calculate cropping values to get from 16x20 to 15x19
    height_crop = (grid_output.shape[1] - 15) // 2
    width_crop = (grid_output.shape[2] - 19) // 2
    
    # Apply precise cropping
    grid_output = Cropping2D(
        cropping=((height_crop, height_crop + (grid_output.shape[1] - 15) % 2),
                 (width_crop, width_crop + (grid_output.shape[2] - 19) % 2)),
        name='grid_output'
    )(grid_output)
    
    response_output = Cropping2D(
        cropping=((height_crop, height_crop + (response_output.shape[1] - 15) % 2),
                 (width_crop, width_crop + (response_output.shape[2] - 19) % 2)),
        name='response_output'
    )(response_output)
    
    print(f"Final grid output shape: {grid_output.shape}")
    print(f"Final response output shape: {response_output.shape}")
    
    # Verify output shapes
    if grid_output.shape[1:3] != (15, 19) or response_output.shape[1:3] != (15, 19):
        raise ValueError(
            f"Invalid output shapes. Expected (15, 19), but got "
            f"grid: {grid_output.shape[1:3]}, response: {response_output.shape[1:3]}"
        )
    
    return Model(inputs, [grid_output, response_output])

def train_unet_with_cross_validation(inputs, before_targets, after_targets, patient_details, model_path, n_folds=5, alpha_values=None):
    """
    Train U-Net model with k-fold cross-validation.
    """
    print(f"Starting {n_folds}-fold cross-validation...")
    
    # Create directory for model checkpoints if it doesn't exist
    model_dir = os.path.dirname(model_path)
    os.makedirs(model_dir, exist_ok=True)
    
    # Initialize k-fold cross-validation
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Initialize lists to store metrics for each fold
    fold_metrics = []
    fold_models = []
    fold_alphas = []
    
    # If no α values are provided, test multiple values
    if alpha_values is None:
        alpha_values = [compute_alpha(class_counts, power=0.5)]
    
    # Prepare data
    '''all_data = {
        'inputs': inputs,
        'after_grids': after_targets['after_grids'],
        'after_responses': after_targets['after_responses'],
        'patient_details': patient_details
    }'''
    
    # Perform k-fold cross-validation
    for fold, (train_idx, val_idx) in enumerate(kfold.split(inputs)):
        print(f"\nFold {fold + 1}/{n_folds}")
        
        # Fold-specific path setup (must go here first!)
        fold_model_dir = os.path.join(model_dir, f"fold_{fold+1}")
        os.makedirs(fold_model_dir, exist_ok=True)
        fold_model_path = os.path.join(fold_model_dir, "model.keras")

        # Validation data
        X_val = inputs[val_idx]
        y_val = {
            'grids': after_targets['after_grids'][val_idx],
            'responses': after_targets['after_responses'][val_idx]
        }

        patient_details_val = [patient_details[i] for i in val_idx]
        
         # If already trained
        if os.path.exists(fold_model_path):
            print(f"Found saved model for Fold {fold + 1}. Loading and evaluating...")
            model = tf.keras.models.load_model(
                fold_model_path,
                compile=False,
                custom_objects={'focal_loss': focal_loss}
            )

            best_alpha = alpha_values[0] if isinstance(alpha_values, list) else alpha_values
            model.compile(
                optimizer='adam',
                loss={'grid_output': focal_loss(alpha=best_alpha), 'response_output': 'mse'},
                metrics={'grid_output': 'accuracy', 'response_output': 'mae'}
            )

            metrics = evaluate_model(model, X_val, y_val, patient_details_val)
            fold_metrics.append(metrics)
            fold_models.append(model)
            fold_alphas.append(best_alpha)
            continue  # Move to next fold

        # Otherwise, train model
        X_train = inputs[train_idx]
        y_train = {
            'grids': after_targets['after_grids'][train_idx],
            'responses': after_targets['after_responses'][train_idx]
        }

        best_dice = -1
        best_alpha = None
        best_model = None
        best_metrics = None
        
        # Test different alpha values
        for alpha in alpha_values:
            print(f"Testing α values: {alpha}")
            
            # Build and compile model
            model = build_unet()
            model.compile(
                optimizer='adam',
                loss={'grid_output': focal_loss(alpha=alpha), 'response_output': 'mse'},
                metrics={'grid_output': 'accuracy', 'response_output': 'mae'}
            )
            
            # Setup callbacks with proper filepath
            fold_model_path = os.path.join(model_dir, f"fold_{fold+1}", "model.keras")
            os.makedirs(os.path.dirname(fold_model_path), exist_ok=True)
            
            checkpoint = ModelCheckpoint(
                fold_model_path,
                monitor='val_loss',
                save_best_only=True,
                mode='min',
                verbose=1
            )
            
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            )
            
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=1e-6,
                verbose=1
            )
            
            # Convert data to tensors
            X_train_tensor = tf.convert_to_tensor(X_train, dtype=tf.float32)
            X_val_tensor = tf.convert_to_tensor(X_val, dtype=tf.float32)
            y_train_grids_tensor = tf.convert_to_tensor(y_train['grids'], dtype=tf.int64)
            y_train_responses_tensor = tf.convert_to_tensor(y_train['responses'], dtype=tf.float32)
            y_val_grids_tensor = tf.convert_to_tensor(y_val['grids'], dtype=tf.int64)
            y_val_responses_tensor = tf.convert_to_tensor(y_val['responses'], dtype=tf.float32)
            
            # Train model
            history = model.fit(
                X_train_tensor,
                {'grid_output': y_train_grids_tensor, 'response_output': y_train_responses_tensor},
                validation_data=(
                    X_val_tensor,
                    {'grid_output': y_val_grids_tensor, 'response_output': y_val_responses_tensor}
                ),
                epochs=150,
                batch_size=8,
                callbacks=[checkpoint, early_stopping, reduce_lr]
            )
            
            # Evaluate model
            metrics = evaluate_model(model, X_val, y_val, patient_details_val)
            dice_score_val = metrics['original_grids']['Dice Score']
            print(f"Fold {fold + 1} - Dice Score for α={alpha}: {dice_score_val}")
            
            # Update best model if better
            if dice_score_val > best_dice:
                best_dice = dice_score_val
                best_alpha = alpha
                best_model = model
                best_metrics = metrics
                
                # Save the best model for this fold
                best_model_path = os.path.join(model_dir, f"best_model_fold_{fold+1}.keras")
                best_model.save(best_model_path)
                print(f"Saved best model for fold {fold + 1} to {best_model_path}")
        
        # Store best results for this fold
        fold_metrics.append(best_metrics)
        fold_models.append(best_model)
        fold_alphas.append(best_alpha)
        
        # Save fold results
        save_fold_results(fold + 1, best_metrics, history.history)
    
    # Calculate and print average metrics across folds
    avg_metrics = calculate_average_metrics(fold_metrics)
    print("\nCross-validation Results:")
    print("\nCross-validation Results:")
    print("Average Validation Metrics across folds:")

    print(f"Grid Accuracy: {avg_metrics['original_grids']['Accuracy']:.4f}")
    print(f"Dice Score: {avg_metrics['original_grids']['Dice Score']:.4f}")
    print(f"Jaccard Index: {avg_metrics['original_grids']['Jaccard Index']:.4f}")

    print(f"Response MAE (Original): {avg_metrics['responses']['Original']['MAE']:.4f}")
    print(f"Response RMSE (Original): {avg_metrics['responses']['Original']['RMSE']:.4f}")
    print(f"No-Response Accuracy (Original): {avg_metrics['responses']['Original']['No-Response Accuracy']:.2%}")

    print(f"Response MAE (Constrained): {avg_metrics['responses']['Constrained']['MAE']:.4f}")
    print(f"Response RMSE (Constrained): {avg_metrics['responses']['Constrained']['RMSE']:.4f}")
    print(f"No-Response Accuracy (Constrained): {avg_metrics['responses']['Constrained']['No-Response Accuracy']:.2%}")
    
    print("\nFinal Test Set Results:")
    # Find the best performing fold
    best_fold_idx = np.argmax([m['original_grids']['Dice Score'] for m in fold_metrics])
    best_model = fold_models[best_fold_idx]
    best_alpha = fold_alphas[best_fold_idx]
    best_metrics = fold_metrics[best_fold_idx]
    
    # Save the overall best model
    best_model.save(model_path)
    print(f"\nSaved overall best model (from fold {best_fold_idx + 1}) to {model_path}")
    
    print(f"\nBest performing fold: {best_fold_idx + 1}")
    print(f"Best α values: {best_alpha}")
    print(f"Best Dice Score: {best_metrics['original_grids']['Dice Score']}")
    
    return best_model, best_metrics, best_alpha, (X_val, y_val, patient_details_val)

def save_fold_results(fold_num, metrics, history):
    """Save metrics and training history for each fold."""
    # Save metrics
    metrics_filename = f"fold_{fold_num}_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    save_metrics_to_csv(metrics, metrics_filename)
    
    # Save training history
    history_df = pd.DataFrame(history)
    history_filename = f"fold_{fold_num}_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    history_df.to_csv(history_filename, index=False)

def calculate_average_metrics(fold_metrics):
    """
    Calculate average metrics across all folds.
    """
    avg_metrics = {
        'original_grids': {},
        'constrained_grids': {},
        'responses': {'Original': {}, 'Constrained': {}}
    }
    
    # Handle grid metrics
    for grid_type in ['original_grids', 'constrained_grids']:
        for metric in ['Accuracy', 'Dice Score', 'Jaccard Index']:
            values = [fold[grid_type][metric] for fold in fold_metrics]
            avg_metrics[grid_type][metric] = np.mean(values)
        
        # Handle per-class metrics
        avg_metrics[grid_type]['Per-class'] = {}
        for class_name in ['Black', 'Grey', 'White']:
            avg_metrics[grid_type]['Per-class'][class_name] = {}
            for metric in ['Precision', 'Recall', 'F1-score']:
                values = [fold[grid_type]['Per-class'][class_name][metric] 
                         for fold in fold_metrics]
                avg_metrics[grid_type]['Per-class'][class_name][metric] = np.mean(values)
    
    # Handle response metrics
    for pred_type in ['Original', 'Constrained']:
        for metric in ['MAE', 'RMSE', 'No-Response Accuracy']:
            values = [fold['responses'][pred_type][metric] for fold in fold_metrics]
            avg_metrics['responses'][pred_type][metric] = np.mean(values)
    
    return avg_metrics

def print_average_metrics(avg_metrics):
    """Print average metrics in a formatted way."""
    print("\nGrid Classification Metrics:")
    print("-" * 40)
    print("Original Predictions:")
    print(f"Average Accuracy: {avg_metrics['original_grids']['Accuracy']:.4f}")
    print(f"Average Dice Score: {avg_metrics['original_grids']['Dice Score']:.4f}")
    print(f"Average Jaccard Index: {avg_metrics['original_grids']['Jaccard Index']:.4f}")
    
    print("\nConstrained Predictions:")
    print(f"Average Accuracy: {avg_metrics['constrained_grids']['Accuracy']:.4f}")
    print(f"Average Dice Score: {avg_metrics['constrained_grids']['Dice Score']:.4f}")
    print(f"Average Jaccard Index: {avg_metrics['constrained_grids']['Jaccard Index']:.4f}")
    
    print("\nPer-class Metrics:")
    for class_name in ['Black', 'Grey', 'White']:
        print(f"\n{class_name} Class:")
        print(f"Original - F1-score: {avg_metrics['original_grids']['Per-class'][class_name]['F1-score']:.4f}")
        print(f"Constrained - F1-score: {avg_metrics['constrained_grids']['Per-class'][class_name]['F1-score']:.4f}")
    
    print("\nResponse Time Metrics:")
    print("-" * 40)
    print(f"Average MAE: {avg_metrics['responses']['MAE']:.4f}")
    print(f"Average RMSE: {avg_metrics['responses']['RMSE']:.4f}")
    # print(f"Average R² Score: {avg_metrics['responses']['R² Score']:.4f}")
    print(f"Average No-Response Accuracy: {avg_metrics['responses']['No-Response Accuracy']:.4f}")

def dice_score(y_true, y_pred, num_classes=3, smooth=1e-6):
    dice_scores = []
    for i in range(num_classes):
        y_true_i = (y_true == i).astype(int)
        y_pred_i = (y_pred == i).astype(int)
        intersection = np.sum(y_true_i * y_pred_i)
        union = np.sum(y_true_i) + np.sum(y_pred_i)
        # Handle edge case: no true/predicted positives
        if union == 0:
            dice_i = 1.0  # Both empty → perfect score
        else:
            dice_i = (2. * intersection + smooth) / (union + smooth)
        
        dice_scores.append(dice_i)
    return np.mean(dice_scores)

def jaccard_index(y_true, y_pred, num_classes=3):
    return jaccard_score(
        y_true.flatten(), 
        y_pred.flatten(), 
        average='weighted', 
        labels=np.arange(num_classes),
        zero_division=0  # Explicitly handle absent classes
    )
       
'''def apply_vision_constraints(before_grid, predicted_grid):
    """
    Apply vision constraints based on before grid state:
    - White (2) must stay White (can't get worse)
    - Grey (1) can only stay Grey or improve to White (can't get worse to Black)
    - Black (0) can improve to any state
    """
    # Ensure both grids have the same shape
    before_grid = before_grid[:15, :19]  # Remove padding if present
    constrained_grid = predicted_grid.copy()
    
    # White pixels must stay white
    white_mask = (before_grid == 2)
    constrained_grid[white_mask] = 2
    
    # Grey pixels can only stay grey or improve to white
    grey_mask = (before_grid == 1)
    constrained_grid[grey_mask & (predicted_grid == 0)] = 1  # Prevent grey→black
    
    return constrained_grid'''

def apply_response_constraints(input_responses, predicted_responses):
    """
    Apply response time constraints:
    - If input response is 0 (no response before treatment), predicted response must be 0
    - Keep other predictions as they are
    
    Args:
        input_responses (numpy array): Input (before treatment) response times
        true_responses (numpy array): True (after treatment) response times
        predicted_responses (numpy array): Predicted response times
        
    Returns:
        numpy array: Constrained predicted response times
    """
    constrained_responses = predicted_responses.copy()
    
    # Create mask for no-response points in input data
    input_no_response_mask = (input_responses == 0)
    
    # Force predicted responses to be 0 where input responses are 0
    constrained_responses[input_no_response_mask] = 0
    
    return constrained_responses

def calculate_class_metrics(y_true, y_pred):
    """
    Calculate per-class metrics.
    """
    metrics = {}
    for class_idx, class_name in enumerate(['Black', 'Grey', 'White']):
        class_mask_true = (y_true == class_idx)
        class_mask_pred = (y_pred == class_idx)
        
        # Calculate metrics for this class
        tp = np.sum(class_mask_true & class_mask_pred)
        fp = np.sum(~class_mask_true & class_mask_pred)
        fn = np.sum(class_mask_true & ~class_mask_pred)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[class_name] = {
            'Precision': precision,
            'Recall': recall,
            'F1-score': f1
        }
    
    return metrics

def evaluate_model(model, X_test, y_test, patient_details_test):
    """
    Evaluate model using real-world response times (unnormalized) and apply vision constraints.
    """
    # Generate predictions
    predictions = model.predict(X_test)
    pred_grids = predictions[0]
    pred_responses = predictions[1]

    # Convert softmax probabilities to class labels
    pred_grids_labels = np.argmax(pred_grids, axis=-1)
    
    # Apply vision constraints
    before_grids = X_test[..., 0]  # First channel contains before grid
    '''constrained_pred_grids = np.array([
        apply_vision_constraints(before_grid[:15, :19], pred_grid)  # Remove padding from before_grid
        for before_grid, pred_grid in zip(before_grids, pred_grids_labels)
    ])'''

    # Extract true labels and input responses
    y_true_grids = y_test['grids']
    y_true_responses = y_test['responses']
    input_responses = X_test[..., 1]  # Second channel contains input responses

    # Get min/max values for each patient
    min_vals = [details['before_min'] for details in patient_details_test]
    max_vals = [details['before_max'] for details in patient_details_test]

    # Unnormalize responses
    unnormalized_pred_responses = unnormalize_responses(pred_responses, min_vals, max_vals)
    unnormalized_true_responses = unnormalize_responses(y_true_responses, min_vals, max_vals)
    unnormalized_input_responses = unnormalize_responses(input_responses[:, :15, :19], min_vals, max_vals)  # Remove padding
    
    # Apply response constraints using input responses
    constrained_pred_responses = apply_response_constraints(
        unnormalized_input_responses,
        unnormalized_pred_responses
    )

    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_true_grids.flatten(), pred_grids_labels.flatten(), labels=[0, 1, 2])

    # Compute metrics for original predictions
    original_metrics = {
        'Accuracy': accuracy_score(y_true_grids.flatten(), pred_grids_labels.flatten()),
        'Dice Score': dice_score(y_true_grids.flatten(), pred_grids_labels.flatten()),
        'Jaccard Index': jaccard_index(y_true_grids, pred_grids_labels),
        'Per-class': calculate_class_metrics(y_true_grids.flatten(), pred_grids_labels.flatten())
    }

    # Compute metrics for constrained predictions
    '''constrained_metrics = {
        'Accuracy': accuracy_score(y_true_grids.flatten(), constrained_pred_grids.flatten()),
        'Dice Score': dice_score(y_true_grids.flatten(), constrained_pred_grids.flatten()),
        'Jaccard Index': jaccard_index(y_true_grids, constrained_pred_grids),
        'Per-class': calculate_class_metrics(y_true_grids.flatten(), constrained_pred_grids.flatten())
    }'''

    # Add response time specific metrics for both original and constrained predictions
    response_metrics = {
        'Original': {
            'MAE': mean_absolute_error(unnormalized_true_responses.flatten(), unnormalized_pred_responses.flatten()),
            'RMSE': np.sqrt(mean_squared_error(unnormalized_true_responses.flatten(), unnormalized_pred_responses.flatten())),
            # 'R² Score': r2_score(unnormalized_true_responses.flatten(), unnormalized_pred_responses.flatten()),
            'No-Response Accuracy': accuracy_score(
                (unnormalized_true_responses.flatten() == 0),
                (unnormalized_pred_responses.flatten() == 0)
            )
        },
        'Constrained': {
            'MAE': mean_absolute_error(unnormalized_true_responses.flatten(), constrained_pred_responses.flatten()),
            'RMSE': np.sqrt(mean_squared_error(unnormalized_true_responses.flatten(), constrained_pred_responses.flatten())),
            # 'R² Score': r2_score(unnormalized_true_responses.flatten(), constrained_pred_responses.flatten()),
            'No-Response Accuracy': 1.0  # Will always be 1.0 due to constraint
        }
    }

    return {
        'original_grids': original_metrics,
        'confusion_matrix': conf_matrix,
        # 'constrained_grids': constrained_metrics,
        'responses': response_metrics
    }

# Visualization updates
def visualize_results(model, X_test, y_test_grid, y_test_response, patient_details, grid_shape=(15, 19), save_folder='v6-predictions'):
    """
    Visualize and save model predictions with patient details, showing both original and constrained predictions.
    Also includes special visualization for input response times.
    """
    print(f"Starting visualization process...")
    os.makedirs(save_folder, exist_ok=True)
    
    # Generate predictions
    pred_grid, pred_response = model.predict(X_test)
    pred_labels = np.argmax(pred_grid, axis=-1)
    
    # Apply constraints
    # before_grids = X_test[..., 0]
    input_responses = X_test[..., 1]  # Get input responses
    '''constrained_pred_labels = np.array([
        apply_vision_constraints(before_grid[:15, :19], pred_grid)
        for before_grid, pred_grid in zip(before_grids, pred_labels)
    ])'''
    
    # Convert tensors to numpy arrays if needed
    X_test = X_test.numpy() if isinstance(X_test, tf.Tensor) else X_test
    y_test_grid = y_test_grid.numpy() if isinstance(y_test_grid, tf.Tensor) else y_test_grid
    y_test_response = y_test_response.numpy() if isinstance(y_test_response, tf.Tensor) else y_test_response

    # Get min/max values for each patient
    min_vals = [details['before_min'] for details in patient_details]
    max_vals = [details['before_max'] for details in patient_details]

    # Unnormalize responses using the new function
    unnormalized_true_response = unnormalize_responses(y_test_response, min_vals, max_vals)
    unnormalized_pred_response = unnormalize_responses(pred_response, min_vals, max_vals)
    unnormalized_input_response = unnormalize_responses(input_responses[:, :15, :19], min_vals, max_vals)  # Remove padding

    # Apply response constraints
    constrained_pred_response = apply_response_constraints(
        unnormalized_input_response,
        #unnormalized_true_response,
        unnormalized_pred_response
    )

    # Create custom colormap for response times
    colors = [(0, 0.7, 0),      # Green (fast responses)
              (1, 1, 0),        # Yellow (medium responses)
              (1, 0, 0)]        # Red (slow responses)
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('response_times', colors, N=n_bins)
    cmap.set_bad(color='black') # No response remains black
   

    for i in range(len(X_test)):
        try:
            # Get patient details
            patient_name, patient_id, eye, date = patient_details[i]['metadata']
            
            # Create figure
            fig, axs = plt.subplots(2, 4, figsize=(24, 12))
            
            # Plot input data (before grid)
            before_grid = X_test[i, :15, :19, 0]
            axs[0,0].imshow(before_grid, cmap='gray')
            axs[0,0].set_title('Before Treatment Grid')
            
            # Plot true after grid
            true_after = y_test_grid[i]
            axs[0,1].imshow(true_after, cmap='gray', vmin=0, vmax=2)
            axs[0,1].set_title('True After Grid')
            
            # Plot predicted grid
            axs[0,2].imshow(pred_labels[i], cmap='gray', vmin=0, vmax=2)
            axs[0,2].set_title('Predicted After Grid')
            
            # Plot constrained predicted grid
            '''axs[0,3].imshow(constrained_pred_labels[i], cmap='gray', vmin=0, vmax=2)
            axs[0,3].set_title('Constrained Predicted Grid')'''
            
            # Plot input response times with masked no-response
            input_response = unnormalized_input_response[i].squeeze()
            input_response_masked = np.ma.masked_where(input_response == 0, input_response)
            im0 = axs[1,0].imshow(input_response_masked.reshape(grid_shape), cmap=cmap)
            axs[1,0].set_title('Input Response Times\n(Black = No Response)')
            plt.colorbar(im0, ax=axs[1,0])
            
            # Plot true response times with masked no-response
            response_true = unnormalized_true_response[i].squeeze()
            response_true_masked = np.ma.masked_where(response_true == 0, response_true)
            im1 = axs[1,1].imshow(response_true_masked.reshape(grid_shape), cmap=cmap)
            axs[1,1].set_title('True Response Times\n(Black = No Response)')
            plt.colorbar(im1, ax=axs[1,1])
            
            # Plot predicted response times with masked no-response
            response_pred = unnormalized_pred_response[i].squeeze()
            response_pred_masked = np.ma.masked_where(response_pred == 0, response_pred)
            im2 = axs[1,2].imshow(response_pred_masked.reshape(grid_shape), cmap=cmap)
            axs[1,2].set_title('Predicted Response Times\n(Black = No Response)')
            plt.colorbar(im2, ax=axs[1,2])
            
            # Plot constrained predicted response times with masked no-response
            response_constrained = constrained_pred_response[i].squeeze()
            response_constrained_masked = np.ma.masked_where(response_constrained == 0, response_constrained)
            im3 = axs[1,3].imshow(response_constrained_masked.reshape(grid_shape), cmap=cmap)
            axs[1,3].set_title('Constrained Predicted Response Times\n(Black = No Response)')
            plt.colorbar(im3, ax=axs[1,3])
            
            # Add super title with patient details
            plt.suptitle(f"Patient: {patient_name}\nID: {patient_id} | Eye: {eye} | Date: {date}", y=1.02)
            
            # Save and close
            filename = f"{patient_id}_{eye.replace(' ', '_')}_{date}.png"
            save_path = os.path.join(save_folder, filename)
            plt.savefig(save_path, bbox_inches='tight')
            plt.close(fig)  # Explicitly close the figure
            
        except Exception as e:
            print(f"Error processing sample {i}: {str(e)}")
            continue

    print(f"Visualization process completed. Check folder: {os.path.abspath(save_folder)}")
    print(f"Number of files in save folder: {len(os.listdir(save_folder))}")

def print_metrics_comparison(final_metrics, csv_metrics=None):
    print("\n" + "="*80)
    print("FINAL TEST SET METRICS (20% Held-out Data)")
    print("="*80)
    
    # Print Final Evaluation Metrics
    print("\n1. FINAL EVALUATION METRICS (Validation Set)")
    print("-"*50)
    
    print("\nA. Predictions:")
    print("  Accuracy:      {:.2%}".format(final_metrics['original_grids']['Accuracy']))
    print("  Dice Score:    {:.2%}".format(final_metrics['original_grids']['Dice Score']))
    print("  Jaccard Index: {:.2%}".format(final_metrics['original_grids']['Jaccard Index']))
    
    print("\n  Per-Class Metrics (Original):")
    print("  {:<10} {:<12} {:<12} {:<12}".format("Class", "Precision", "Recall", "F1-score"))
    print("  " + "-"*46)
    for class_name, metrics in final_metrics['original_grids']['Per-class'].items():
        print("  {:<10} {:<12.2%} {:<12.2%} {:<12.2%}".format(
            class_name,
            metrics['Precision'],
            metrics['Recall'],
            metrics['F1-score']
        ))
    
    '''print("\nB. Constrained Predictions:")
    print("  Accuracy:      {:.2%}".format(final_metrics['constrained_grids']['Accuracy']))
    print("  Dice Score:    {:.2%}".format(final_metrics['constrained_grids']['Dice Score']))
    print("  Jaccard Index: {:.2%}".format(final_metrics['constrained_grids']['Jaccard Index']))
    
    print("\n  Per-Class Metrics (Constrained):")
    print("  {:<10} {:<12} {:<12} {:<12}".format("Class", "Precision", "Recall", "F1-score"))
    print("  " + "-"*46)
    for class_name, metrics in final_metrics['constrained_grids']['Per-class'].items():
        print("  {:<10} {:<12.2%} {:<12.2%} {:<12.2%} ".format(
            class_name,
            metrics['Precision'],
            metrics['Recall'],
            metrics['F1-score']
        ))'''
    
    print("\nC. Response Time Metrics:")
    print("-" * 40)
    print("Original Predictions:")
    print("  MAE:                  {:.2f}".format(final_metrics['responses']['Original']['MAE']))
    print("  RMSE:                 {:.2f}".format(final_metrics['responses']['Original']['RMSE']))
    # print(f"Average R² Score: {final_metrics['responses']['Original']['R² Score']:.4f}")
    print("  No-Response Accuracy: {:.2%}".format(final_metrics['responses']['Original']['No-Response Accuracy']))
    
    print("\nConstrained Predictions:")
    print("  MAE:                  {:.2f}".format(final_metrics['responses']['Constrained']['MAE']))
    print("  RMSE:                 {:.2f}".format(final_metrics['responses']['Constrained']['RMSE']))
    # print(f"Constrained R² Score: {final_metrics['responses']['Constrained']['R² Score']:.4f}")
    print("  No-Response Accuracy: {:.2%}".format(final_metrics['responses']['Constrained']['No-Response Accuracy']))
    
    # Print CSV Metrics if available
    if csv_metrics:
        print("\n2. TEST SET METRICS")
        print("-"*50)
        print("  {:<20} {:<15}".format("Metric", "Original"))
        print("  " + "-"*50)
        print("  {:<20} {:<15.2%}".format(
            "Accuracy",
            csv_metrics['original_accuracy']
        ))
        print("  {:<20} {:<15.2%}".format(
            "Dice Score",
            csv_metrics['original_dice']
        ))
        
        print("\nB. Per-Class F1-Scores:")
        print("  {:<10} {:<15}".format("Class", "Original"))
        print("  " + "-"*40)
        for class_name in ['Black', 'Grey', 'White']:
            print("  {:<10} {:<15.2%}".format(
                class_name,
                csv_metrics['original_f1'][class_name]
            ))
        
        print("\nC. Response Time Metrics:")
        print("  {:<20} {:<15} {:<15}".format("Metric", "Original", "Constrained"))
        print("  " + "-"*50)
        print("  {:<20} {:<15.2f} {:<15.2f}".format(
            "MAE",
            final_metrics['responses']['Original']['MAE'],
            final_metrics['responses']['Constrained']['MAE']
        ))
        print("  {:<20} {:<15.2f} {:<15.2f}".format(
            "RMSE",
            final_metrics['responses']['Original']['RMSE'],
            final_metrics['responses']['Constrained']['RMSE']
        ))
        # print("  {:<20} {:<15.4f} {:<15.4f}".format(
        #     "R² Score",
        #     final_metrics['responses']['Original']['R² Score'],
        #     final_metrics['responses']['Constrained']['R² Score']
        # ))
        print("  {:<20} {:<15.2%} {:<15.2%}".format(
            "No-Response Acc.",
            final_metrics['responses']['Original']['No-Response Accuracy'],
            final_metrics['responses']['Constrained']['No-Response Accuracy']
        ))
    
    print("\n" + "="*80)

def save_metrics_to_csv(metrics, filename=None):
    """
    Save evaluation metrics to a CSV file with improved logging.
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"model_metrics_{timestamp}.csv"
    
    # Create metrics dictionary
    metrics_dict = {}
    
    # Add original grid metrics
    for key, value in metrics['original_grids'].items():
        if key != 'Per-class':
            metrics_dict[f'Original_{key}'] = value
    
    # Add per-class metrics for original predictions
    for class_name, class_metrics in metrics['original_grids']['Per-class'].items():
        for metric_name, metric_value in class_metrics.items():
            metrics_dict[f'Original_{class_name}_{metric_name}'] = metric_value
    
    '''# Add constrained grid metrics
    for key, value in metrics['constrained_grids'].items():
        if key != 'Per-class':
            metrics_dict[f'Constrained_{key}'] = value
    
    # Add per-class metrics for constrained predictions
    for class_name, class_metrics in metrics['constrained_grids']['Per-class'].items():
        for metric_name, metric_value in class_metrics.items():
            metrics_dict[f'Constrained_{class_name}_{metric_name}'] = metric_value'''
    
    # Add response metrics
    for key, value in metrics['responses']['Original'].items():
        metrics_dict[f'Original_{key}'] = value
    for key, value in metrics['responses']['Constrained'].items():
        metrics_dict[f'Constrained_{key}'] = value
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame([metrics_dict])
    df.to_csv(filename, index=False)
    print(f"\nMetrics saved to: {filename}")
    
    # Extract metrics for comparison
    csv_metrics = {
        'original_accuracy': metrics['original_grids']['Accuracy'],
        #'constrained_accuracy': metrics['constrained_grids']['Accuracy'],
        'original_dice': metrics['original_grids']['Dice Score'],
        #'constrained_dice': metrics['constrained_grids']['Dice Score'],
        'original_f1': {
            class_name: metrics['original_grids']['Per-class'][class_name]['F1-score']
            for class_name in ['Black', 'Grey', 'White']
        },
        #'constrained_f1': {
        #    class_name: metrics['constrained_grids']['Per-class'][class_name]['F1-score']
        #    for class_name in ['Black', 'Grey', 'White']
        #}
    }
    
    # Print detailed comparison
    print_metrics_comparison(metrics, csv_metrics)

def generate_gradcam_visualization(model, input_image, layer_name='grid_output', class_idx=None):
    """
    Generate and visualize Grad-CAM heatmaps for the U-Net model.
    
    Args:
        model: Trained U-Net model
        input_image: Single input image (should be shape [1, 16, 20, 2])
        layer_name: Name of the target layer (default is final grid output)
        class_idx: Index of class to explain (0=Black, 1=Grey, 2=White)
    
    Returns:
        tuple: (original_image, heatmap, superimposed_image)
    """
    # Ensure input is a batch
    if len(input_image.shape) == 3:
        input_image = np.expand_dims(input_image, axis=0)
    
    # Create a model that maps the input to the target layer and final output
    grad_model = Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output[0]]
    )
    
    # Record operations for automatic differentiation
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_image)
        if class_idx is None:
            class_idx = tf.argmax(predictions[0], axis=-1)
        class_output = predictions[:, :, :, class_idx]
    
    # Gradient of the output with respect to the last conv layer
    grads = tape.gradient(class_output, conv_outputs)
    
    # Vector of mean intensity of gradients over feature map
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Multiply each channel by importance and sum
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    
    # ReLU and normalize
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()

def visualize_gradcam_results(model, X_test, y_test, save_folder='gradcam_results'):
    """
    Generate and save Grad-CAM visualizations for test samples.
    """
    os.makedirs(save_folder, exist_ok=True)
    
    # Update target layers to match actual model layer names
    target_layers = [
        'grid_output',
        'conv2d_136',
        'conv2d_154',
        'conv2d_174'
    ]
    
    class_names = ['Black', 'Grey', 'White']
    
    for idx in range(min(5, len(X_test))):
        input_image = X_test[idx:idx+1]
        true_grid = y_test['grids'][idx]
        original_grid = input_image[0, :, :, 0]
        
        # First figure: Class-specific Grad-CAM
        plt.figure(figsize=(20, 10))
        
        # Original inputs
        plt.subplot(2, 4, 1)
        plt.imshow(original_grid, cmap='gray')
        plt.title('Input Grid')
        plt.axis('off')
        
        plt.subplot(2, 4, 2)
        plt.imshow(input_image[0, :, :, 1], cmap='viridis')
        plt.title('Input Response Times')
        plt.colorbar()
        plt.axis('off')
        
        # True and predicted grids
        plt.subplot(2, 4, 3)
        plt.imshow(true_grid, cmap='gray')
        plt.title('True Grid')
        plt.axis('off')
        
        pred = model.predict(input_image)
        pred_grid = np.argmax(pred[0], axis=-1)
        
        plt.subplot(2, 4, 4)
        plt.imshow(pred_grid[0], cmap='gray')
        plt.title('Predicted Grid')
        plt.axis('off')
        
        # Class-specific Grad-CAM
        for i, class_name in enumerate(class_names):
            plt.subplot(2, 4, i+5)
            heatmap = generate_gradcam_visualization(
                model, 
                input_image,
                layer_name='grid_output',
                class_idx=i
            )
            plt.imshow(heatmap, cmap='jet', alpha=0.5)
            plt.title(f'Grad-CAM: {class_name}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, f'gradcam_sample_{idx}.png'))
        plt.close()
        
        # Generate layer-wise Grad-CAM
        plt.figure(figsize=(15, 5))
        for j, layer in enumerate(target_layers):
            plt.subplot(1, 4, j+1)
            try:
                heatmap = generate_gradcam_visualization(
                    model,
                    input_image,
                    layer_name=layer,
                    class_idx=np.argmax(pred_grid[0].flatten())
                )
                
                plt.imshow(input_image[0, :, :, 0], cmap='gray')
                plt.imshow(heatmap, cmap='jet', alpha=0.5)
                plt.title(f'Layer: {layer}')
            except Exception as e:
                print(f"Warning: Could not generate Grad-CAM for layer {layer}: {str(e)}")
                plt.title(f'Layer: {layer}\n(Failed)')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, f'gradcam_layers_sample_{idx}.png'))
        plt.close()

    print(f"Grad-CAM visualizations saved to: {save_folder}")

def generate_explainability_visualizations(model, X_test, y_test, patient_details, save_folder='explainability_results'):
    """
    Generate comprehensive explainability visualizations including:
    - Feature importance maps
    - Layer activations
    - Attention maps
    - Decision boundaries
    """
    os.makedirs(save_folder, exist_ok=True)
    print("Generating explainability visualizations...")

    # Process a subset of test samples
    for idx in range(min(5, len(X_test))):
        input_image = X_test[idx:idx+1]
        true_grid = y_test['grids'][idx]
        true_response = y_test['responses'][idx]
        patient_info = patient_details[idx]['metadata']

        # Create a large figure for all visualizations
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Original Inputs and Predictions
        plt.subplot(3, 4, 1)
        plt.imshow(input_image[0, :, :, 0], cmap='gray')
        plt.title('Input Grid')
        plt.axis('off')

        plt.subplot(3, 4, 2)
        plt.imshow(input_image[0, :, :, 1], cmap='viridis')
        plt.title('Input Response Times')
        plt.colorbar()
        plt.axis('off')

        # Get model predictions
        pred = model.predict(input_image)
        pred_grid = np.argmax(pred[0], axis=-1)
        
        plt.subplot(3, 4, 3)
        plt.imshow(true_grid, cmap='gray')
        plt.title('True Grid')
        plt.axis('off')

        plt.subplot(3, 4, 4)
        plt.imshow(pred_grid[0], cmap='gray')
        plt.title('Predicted Grid')
        plt.axis('off')

        # 2. Feature Importance for each class
        class_names = ['Black', 'Grey', 'White']
        for i, class_name in enumerate(class_names):
            plt.subplot(3, 4, i+5)
            heatmap = generate_gradcam_visualization(
                model, 
                input_image,
                layer_name='grid_output',
                class_idx=i
            )
            plt.imshow(input_image[0, :, :, 0], cmap='gray')
            plt.imshow(heatmap, cmap='jet', alpha=0.5)
            plt.title(f'Class Importance: {class_name}')
            plt.axis('off')

        # 3. Attention Maps
        # Get intermediate attention layers
        attention_layers = [layer for layer in model.layers if 'attention' in layer.name.lower()]
        for i, layer in enumerate(attention_layers[:3]):  # Show first 3 attention maps
            plt.subplot(3, 4, i+8)
            attention_model = Model(inputs=model.input, outputs=layer.output)
            attention_map = attention_model.predict(input_image)
            plt.imshow(attention_map[0, ..., 0], cmap='viridis')
            plt.title(f'Attention Map {i+1}')
            plt.axis('off')

        # 4. Decision Confidence
        plt.subplot(3, 4, 11)
        confidence_map = np.max(pred[0][0], axis=-1)
        plt.imshow(confidence_map, cmap='RdYlGn')
        plt.title('Prediction Confidence')
        plt.colorbar()
        plt.axis('off')

        # 5. Error Analysis
        plt.subplot(3, 4, 12)
        error_map = (pred_grid[0] != true_grid).astype(float)
        plt.imshow(error_map, cmap='RdYlGn_r')
        plt.title('Prediction Errors')
        plt.colorbar()
        plt.axis('off')

        # Add patient information as super title
        patient_name, patient_id, eye, date = patient_info
        plt.suptitle(
            f"Explainability Analysis\n"
            f"Patient: {patient_name} | ID: {patient_id}\n"
            f"Eye: {eye} | Date: {date}",
            y=1.02
        )

        # Save the figure
        plt.tight_layout()
        filename = f"explainability_{patient_id}_{eye.replace(' ', '_')}_{date}.png"
        plt.savefig(os.path.join(save_folder, filename), bbox_inches='tight', dpi=300)
        plt.close()

    print(f"Explainability visualizations saved to: {save_folder}")

def plot_confusion_matrix(conf_matrix, class_labels=['Black', 'Grey', 'White'], save_path=None):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    else:
        plt.show()
        
# Main function
def main():
    # Load/prepare data
    print("\nLoading and preparing dataset...")
    padded_inputs, before_targets, after_targets, patient_details, before_grid_files = prepare_dataset()
    
    # Train with cross-validation or load existing model
    if os.path.exists(model_path):
        print(f"\nFound existing model at {model_path}. Loading model...")
        best_alpha = [1.737723729905418, 2.67520315251901, 1.3747597256737596]
        model = tf.keras.models.load_model(
            model_path,
            compile=False,
            custom_objects={'focal_loss': focal_loss}
        )
        model.compile(
            optimizer='adam',
            loss={'grid_output': focal_loss(alpha=best_alpha), 'response_output': 'mse'},
            metrics={'grid_output': 'accuracy', 'response_output': 'mae'}
        )
        print("Model loaded successfully.")
        
        # Use a portion of data for testing
        print("\nSplitting data for testing...")
        _, X_test, _, y_test_grids, _, y_test_responses, _, test_patient_details = train_test_split(
            padded_inputs, after_targets['after_grids'], after_targets['after_responses'],
            patient_details, test_size=0.2, random_state=42
        )
        y_test = {'grids': y_test_grids, 'responses': y_test_responses}
        
        print("\nEvaluating model on test set...")
        metrics = evaluate_model(model, X_test, y_test, test_patient_details)
        
        plot_confusion_matrix(
        metrics['confusion_matrix'],
        save_path='confusion_matrix_unet.png'
        )

        # Print detailed metrics including response metrics
        print("\nDETAILED EVALUATION METRICS")
        print("=" * 50)
        
        print("\nGrid Classification Metrics:")
        print("-" * 40)
        print(f"Original Accuracy: {metrics['original_grids']['Accuracy']:.4f}")
        print(f"Original Dice Score: {metrics['original_grids']['Dice Score']:.4f}")
        print(f"Original Jaccard Index: {metrics['original_grids']['Jaccard Index']:.4f}")
        
        print("\nPer-class Metrics:")
        print("  {:<10} {:<12} {:<12} {:<12}".format("Class", "Precision", "Recall", "F1-score"))
        print("  " + "-" * 46)
        for class_name, class_metrics in metrics['original_grids']['Per-class'].items():
            print("  {:<10} {:<12.4%} {:<12.4%} {:<12.4%}".format(
                class_name,
                class_metrics['Precision'],
                class_metrics['Recall'],
                class_metrics['F1-score']
            ))
        
        print("\nResponse Time Metrics:")
        print("-" * 40)
        print(f"Original MAE: {metrics['responses']['Original']['MAE']:.4f}")
        print(f"Original RMSE: {metrics['responses']['Original']['RMSE']:.4f}")
        # print(f"Original R² Score: {metrics['responses']['Original']['R² Score']:.4f}")
        print(f"Original No-Response Accuracy: {metrics['responses']['Original']['No-Response Accuracy']:.4f}")
        
        print("\nConstrained Response Time Metrics:")
        print("-" * 40)
        print(f"Constrained MAE: {metrics['responses']['Constrained']['MAE']:.4f}")
        print(f"Constrained RMSE: {metrics['responses']['Constrained']['RMSE']:.4f}")
        # print(f"Constrained R² Score: {metrics['responses']['Constrained']['R² Score']:.4f}")
        print(f"Constrained No-Response Accuracy: {metrics['responses']['Constrained']['No-Response Accuracy']:.4f}")
    else:
        print("\nNo existing model found. Training with cross-validation...")
        model, metrics, best_alpha, (X_test, y_test, test_patient_details) = train_unet_with_cross_validation(
            padded_inputs, before_targets, after_targets, patient_details, model_path
        )
    
    print("\nBest α values used:", best_alpha)
    
    # Save metrics and visualize results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_metrics_to_csv(metrics, f"final_metrics_{timestamp}.csv")
    
    print("\nGenerating visualizations...")
    visualize_results(
        model=model,
        X_test=X_test,
        y_test_grid=y_test['grids'],
        y_test_response=y_test['responses'],
        patient_details=test_patient_details
    )
    
    print("\nGenerating explainability visualizations...")
    generate_explainability_visualizations(
        model=model,
        X_test=X_test,
        y_test=y_test,
        patient_details=test_patient_details
    )

    visualize_gradcam_results(
        model=model,
        X_test=X_test,
        y_test=y_test
    )
    
    print("\nProcess completed successfully!")

if __name__ == "__main__":
    main()