# Support Vector Machine (SVM) Algorithm Documentation

## KOI Exoplanet Detection Implementation

---

## Table of Contents

1. [Overview](#overview)
2. [Algorithm Theory](#algorithm-theory)
3. [Dataset and Features](#dataset-and-features)
4. [Implementation Details](#implementation-details)
5. [Data Preprocessing](#data-preprocessing)
6. [Model Architecture](#model-architecture)
7. [Hyperparameter Tuning](#hyperparameter-tuning)
8. [Performance Evaluation](#performance-evaluation)
9. [Model Deployment](#model-deployment)
10. [Usage Guide](#usage-guide)

---

## Overview

This documentation describes the implementation of a Support Vector Machine (SVM) classifier for detecting exoplanet candidates using the Kepler Objects of Interest (KOI) dataset. The model classifies celestial objects as either confirmed exoplanet candidates or non-candidates based on various astronomical features.

### Key Objectives:

- Build a robust binary classifier for exoplanet detection
- Achieve high accuracy and reliability in predictions
- Provide probability estimates for classification confidence
- Create a deployable model for real-world astronomical applications

---

## Algorithm Theory

### What is Support Vector Machine (SVM)?

Support Vector Machine is a powerful supervised machine learning algorithm used for both classification and regression tasks. For our exoplanet detection problem, we use SVM for binary classification.

### Core Principles:

1. **Optimal Hyperplane**: SVM finds the optimal decision boundary (hyperplane) that separates the two classes with maximum margin.

2. **Support Vectors**: These are the data points closest to the decision boundary. They are critical in defining the hyperplane.

3. **Kernel Trick**: SVM can handle non-linearly separable data by mapping it to higher-dimensional space using kernel functions.

### Mathematical Foundation:

The SVM optimization problem aims to minimize:

```
minimize: (1/2)||w||² + C∑ξᵢ
subject to: yᵢ(w·xᵢ + b) ≥ 1 - ξᵢ
```

Where:

- `w` = weight vector (defines the hyperplane)
- `C` = regularization parameter (controls trade-off between margin and misclassification)
- `ξᵢ` = slack variables (allow some misclassification)
- `b` = bias term

### Kernel Functions Used:

1. **Linear Kernel**: `K(xᵢ, xⱼ) = xᵢ · xⱼ`

   - Best for linearly separable data
   - Provides interpretable feature importance

2. **RBF (Radial Basis Function) Kernel**: `K(xᵢ, xⱼ) = exp(-γ||xᵢ - xⱼ||²)`
   - Handles non-linear patterns
   - Most commonly effective kernel
   - γ parameter controls the influence of individual training examples

---

## Dataset and Features

### Dataset Source

- **Name**: Kepler Objects of Interest (KOI) Playground Dataset
- **Source**: NASA Kepler Space Telescope observations
- **Purpose**: Training data for exoplanet detection algorithms

### Target Variable

- **Variable**: `koi_disposition`
- **Classes**:
  - `candidate`: Confirmed or likely exoplanet
  - `non-candidate`: False positive or non-exoplanet object

### Feature Description

Our SVM model uses the following 15 astronomical features:

| Feature            | Description                   | Unit/Range        | Importance                         |
| ------------------ | ----------------------------- | ----------------- | ---------------------------------- |
| `kepid`            | Kepler Input Catalog ID       | Unique identifier | Identifier only                    |
| `koi_period`       | Orbital period                | Days              | **High** - Primary indicator       |
| `koi_time0bk`      | Time of first transit         | BKJD              | Medium                             |
| `koi_impact`       | Impact parameter              | 0-1               | **High** - Orbit geometry          |
| `koi_duration`     | Transit duration              | Hours             | **High** - Transit characteristics |
| `koi_depth`        | Transit depth                 | ppm               | **Very High** - Signal strength    |
| `koi_incl`         | Orbital inclination           | Degrees           | Medium                             |
| `koi_model_snr`    | Signal-to-noise ratio         | Ratio             | **Very High** - Data quality       |
| `koi_count`        | Number of transits            | Count             | Medium                             |
| `koi_bin_oedp_sig` | Binary discrimination test    | 0-1               | High                               |
| `koi_steff`        | Stellar effective temperature | Kelvin            | Medium                             |
| `koi_slogg`        | Stellar surface gravity       | log₁₀(cm/s²)      | Medium                             |
| `koi_srad`         | Stellar radius                | Solar radii       | Medium                             |
| `koi_smass`        | Stellar mass                  | Solar masses      | Medium                             |
| `koi_kepmag`       | Kepler magnitude              | Magnitude         | Low                                |

### Feature Categories:

1. **Transit Properties** (Most Important):

   - Period, duration, depth, impact parameter
   - Direct indicators of exoplanet presence

2. **Signal Quality**:

   - SNR, binary discrimination
   - Measure data reliability

3. **Stellar Properties**:
   - Temperature, gravity, radius, mass
   - Context for planet detection

---

## Implementation Details

### Technology Stack

- **Language**: Python 3.x
- **Primary Library**: scikit-learn
- **Data Handling**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Model Persistence**: joblib

### Code Structure

```
KOI_Exoplanet_SVM_Analysis.ipynb
├── Data Loading and Exploration
├── Feature Analysis and Visualization
├── Data Preprocessing
├── Train-Test Split
├── Hyperparameter Tuning
├── Model Training and Evaluation
├── Performance Visualization
├── Final Model Training
└── Model Export and Deployment
```

---

## Data Preprocessing

### 1. Missing Value Handling

```python
# Numerical features: filled with median
for col in numerical_columns:
    df[col].fillna(df[col].median(), inplace=True)

# Categorical features: filled with mode
for col in categorical_columns:
    df[col].fillna(df[col].mode()[0], inplace=True)
```

**Strategy**:

- Median imputation preserves distribution for numerical features
- Mode imputation maintains most common category for categorical features

### 2. Feature Encoding

```python
# Categorical variables
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Target variable
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)
```

**Purpose**: Convert categorical variables to numerical format for SVM processing

### 3. Feature Scaling

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

**Method**: StandardScaler (z-score normalization)

- Formula: `z = (x - μ) / σ`
- **Why Critical**: SVM is distance-based and sensitive to feature scales
- **Result**: All features have mean=0, standard deviation=1

### 4. Data Splitting Strategy

```python
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)
```

**Configuration**:

- 80% training, 20% testing
- Stratified split maintains class balance
- Fixed random state ensures reproducibility

---

## Model Architecture

### SVM Configuration

```python
SVC(
    C=best_C,                    # Regularization parameter
    kernel=best_kernel,          # 'rbf' or 'linear'
    gamma=best_gamma,            # Kernel coefficient (for RBF)
    probability=True,            # Enable probability predictions
    random_state=42              # Reproducibility
)
```

### Key Parameters:

1. **C (Regularization Parameter)**

   - **Range Tested**: [10, 100, 1000]
   - **Purpose**: Controls trade-off between margin maximization and training error
   - **High C**: Lower bias, higher variance (risk of overfitting)
   - **Low C**: Higher bias, lower variance (risk of underfitting)

2. **Kernel**

   - **Options Tested**: ['rbf', 'linear']
   - **RBF**: Handles non-linear relationships
   - **Linear**: Provides interpretable feature weights

3. **Gamma (for RBF kernel)**
   - **Range Tested**: ['scale', 0.01, 0.1]
   - **Purpose**: Controls influence radius of support vectors
   - **High γ**: Tight fit to training data (overfitting risk)
   - **Low γ**: Smooth decision boundary (underfitting risk)

---

## Hyperparameter Tuning

### Grid Search Configuration

```python
param_grid = {
    'C': [10, 100, 1000],
    'kernel': ['rbf', 'linear'],
    'gamma': ['scale', 0.01, 0.1]
}

grid_search = GridSearchCV(
    svm, param_grid,
    cv=3,                        # 3-fold cross-validation
    scoring='accuracy',          # Optimization metric
    n_jobs=-1,                   # Use all CPU cores
    verbose=2,                   # Progress tracking
    return_train_score=True      # Include training scores
)
```

### Search Space:

- **Total Combinations**: 18 (3 × 2 × 3)
- **Cross-Validation**: 3-fold CV for robust evaluation
- **Scoring Metric**: Accuracy (suitable for balanced dataset)

### Optimization Process:

1. **Exhaustive Search**: Tests all parameter combinations
2. **Cross-Validation**: Each combination evaluated on 3 different train/validation splits
3. **Best Model Selection**: Highest average CV accuracy
4. **Final Training**: Best parameters used to train on full dataset

---

## Performance Evaluation

### Evaluation Metrics

1. **Accuracy**: `(TP + TN) / (TP + TN + FP + FN)`

   - Overall correctness of predictions

2. **Precision**: `TP / (TP + FP)`

   - Proportion of positive predictions that are correct
   - Critical for avoiding false exoplanet detections

3. **Recall (Sensitivity)**: `TP / (TP + FN)`

   - Proportion of actual positives correctly identified
   - Important for not missing real exoplanets

4. **F1-Score**: `2 × (Precision × Recall) / (Precision + Recall)`

   - Harmonic mean of precision and recall
   - Balances both metrics

5. **ROC-AUC**: Area Under ROC Curve
   - Measures model's ability to distinguish between classes
   - Values closer to 1.0 indicate better performance

### Confusion Matrix Analysis

```
                 Predicted
                 0    1
Actual    0    [TN] [FP]
          1    [FN] [TP]
```

### Probability Calibration

- Model provides probability estimates for each prediction
- Calibration curves assess probability reliability
- Well-calibrated probabilities enable confidence thresholds

---

## Model Deployment

### Model Persistence

The trained model and preprocessing components are saved using joblib:

```python
# Model artifacts saved:
├── koi_svm_model_[timestamp].joblib          # Trained SVM model
├── koi_scaler_[timestamp].joblib             # Feature scaler
├── koi_target_encoder_[timestamp].joblib     # Target label encoder
├── koi_label_encoders_[timestamp].joblib     # Categorical encoders
└── koi_model_metadata_[timestamp].joblib     # Model metadata
```

### Model Metadata

```python
metadata = {
    'model_type': 'SVM',
    'best_parameters': {...},           # Optimal hyperparameters
    'cv_score': float,                  # Cross-validation accuracy
    'training_accuracy': float,         # Final training accuracy
    'feature_names': [...],             # Input feature names
    'target_classes': [...],            # Output class labels
    'training_samples': int,            # Dataset size
    'timestamp': str,                   # Creation timestamp
    'model_files': {...}                # File paths
}
```

---

## Usage Guide

### Loading the Model

```python
import joblib
import pandas as pd
import numpy as np

# Load model components
model = joblib.load('models/koi_svm_model_[timestamp].joblib')
scaler = joblib.load('models/koi_scaler_[timestamp].joblib')
target_encoder = joblib.load('models/koi_target_encoder_[timestamp].joblib')
metadata = joblib.load('models/koi_model_metadata_[timestamp].joblib')
```

### Making Predictions

```python
def predict_exoplanet(sample_features):
    """
    Predict exoplanet probability for new astronomical data

    Parameters:
    -----------
    sample_features : dict or pandas.Series
        Dictionary or Series containing feature values with keys:
        ['koi_period', 'koi_time0bk', 'koi_impact', 'koi_duration',
         'koi_depth', 'koi_incl', 'koi_model_snr', 'koi_count',
         'koi_bin_oedp_sig', 'koi_steff', 'koi_slogg', 'koi_srad',
         'koi_smass', 'koi_kepmag']

    Returns:
    --------
    dict : Prediction results containing:
        - 'predicted_class': 'candidate' or 'non-candidate'
        - 'probability_candidate': float [0-1]
        - 'probability_non_candidate': float [0-1]
        - 'confidence': float [0-1] (max probability)
    """

    # Convert to DataFrame
    if isinstance(sample_features, dict):
        sample_df = pd.DataFrame([sample_features])
    else:
        sample_df = pd.DataFrame([sample_features])

    # Apply preprocessing (scaling)
    sample_scaled = scaler.transform(sample_df)

    # Make prediction
    prediction = model.predict(sample_scaled)[0]
    probabilities = model.predict_proba(sample_scaled)[0]

    # Convert back to original labels
    predicted_class = target_encoder.inverse_transform([prediction])[0]

    return {
        'predicted_class': predicted_class,
        'probability_non_candidate': probabilities[0],
        'probability_candidate': probabilities[1],
        'confidence': max(probabilities)
    }
```

### Example Usage

```python
# Example: Predict for a new KOI object
new_observation = {
    'koi_period': 365.25,        # Earth-like orbital period
    'koi_time0bk': 140.0,        # Time of first transit
    'koi_impact': 0.5,           # Impact parameter
    'koi_duration': 6.5,         # Transit duration (hours)
    'koi_depth': 100.0,          # Transit depth (ppm)
    'koi_incl': 89.5,            # Orbital inclination
    'koi_model_snr': 15.0,       # Signal-to-noise ratio
    'koi_count': 4,              # Number of transits observed
    'koi_bin_oedp_sig': 0.8,     # Binary discrimination
    'koi_steff': 5778,           # Stellar temperature (K)
    'koi_slogg': 4.44,           # Stellar surface gravity
    'koi_srad': 1.0,             # Stellar radius (solar radii)
    'koi_smass': 1.0,            # Stellar mass (solar masses)
    'koi_kepmag': 12.0           # Kepler magnitude
}

# Make prediction
result = predict_exoplanet(new_observation)

print(f"Predicted class: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.3f}")
print(f"Probability of being candidate: {result['probability_candidate']:.3f}")
```

### Interpretation Guidelines

**High Confidence Predictions (>0.9)**:

- Reliable classifications
- Safe for automated processing

**Medium Confidence Predictions (0.7-0.9)**:

- Good predictions
- May benefit from additional verification

**Low Confidence Predictions (<0.7)**:

- Uncertain classifications
- Recommend manual review or additional observations

---

## Key Advantages of This SVM Implementation

1. **Robust Classification**: SVM's margin maximization provides good generalization
2. **Probability Estimates**: Enables confidence-based decision making
3. **Feature Scaling**: Proper preprocessing ensures optimal performance
4. **Hyperparameter Optimization**: Grid search finds best configuration
5. **Comprehensive Evaluation**: Multiple metrics assess different aspects
6. **Deployable Model**: Complete pipeline ready for production use

---

## Limitations and Considerations

1. **Computational Complexity**: O(n²) to O(n³) training time
2. **Memory Requirements**: Stores support vectors (can be large)
3. **Parameter Sensitivity**: Requires careful hyperparameter tuning
4. **Feature Scaling Dependency**: Preprocessing is critical
5. **Interpretability**: RBF kernel provides limited feature importance

---

## Future Improvements

1. **Advanced Kernels**: Experiment with polynomial or custom kernels
2. **Feature Engineering**: Create derived astronomical features
3. **Ensemble Methods**: Combine with other algorithms
4. **Online Learning**: Implement incremental updates
5. **Deep Learning**: Compare with neural network approaches

---

_This documentation provides a comprehensive guide to understanding and using the SVM-based exoplanet detection system. For additional technical details, refer to the accompanying Jupyter notebook implementation._
