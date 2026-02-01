# Predictive Maintenance Analysis

This project implements a comprehensive predictive maintenance solution using machine learning techniques to predict equipment failures before they occur.

## Overview

The project analyzes industrial equipment sensor data to predict machine failures and classify different types of failures. It includes data preprocessing, exploratory data analysis, feature engineering using PCA, and multiple machine learning models.

## Dataset

The project uses the AI4I 2020 Predictive Maintenance Dataset which contains:
- 10,000 data points
- Sensor readings: Air temperature, Process temperature, Rotational speed, Torque, Tool wear
- Machine quality variants: Low (L), Medium (M), High (H)
- Failure types: Tool Wear Failure (TWF), Heat Dissipation Failure (HDF), Power Failure (PWF), Overstrain Failure (OSF)

## Key Features

### Data Analysis
- **Exploratory Data Analysis (EDA)**: Comprehensive data exploration and visualization
- **Statistical Testing**: Chi-square tests and Mutual Information analysis
- **Data Cleaning**: Handling inconsistencies and outliers

### Feature Engineering
- **SMOTENC Oversampling**: Handling class imbalance for both categorical and numerical features
- **Principal Component Analysis (PCA)**: Dimensionality reduction with meaningful component interpretation:
  - **Thermal Stress Index**: Captures temperature and torque relationships
  - **Speed vs Force**: Mechanical operating mode variations
  - **Tool Health Index**: Independent tool wear assessment

### Machine Learning Models

Three optimized models were implemented with hyperparameter tuning:

1. **Logistic Regression**
   - Hyperopt optimization for C, solver, and penalty parameters
   - Excellent baseline performance

2. **Random Forest**
   - Optimized n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features
   - Strong feature importance insights

3. **XGBoost**
   - Advanced gradient boosting with learning_rate, subsample, colsample_bytree optimization
   - Superior performance with excellent precision-recall characteristics

### Model Performance

All models achieved excellent performance:
- **Precision**: >99% for failure detection
- **Recall**: >99% for catching actual failures  
- **Average Precision (AP)**: ~0.99 indicating near-perfect precision-recall trade-off
- **ROC-AUC**: >0.99 showing excellent overall classification ability

## Key Insights

1. **Tool Wear** is the most critical independent predictor of failure
2. **Thermal Stress** (temperature + torque) creates distinct failure patterns
3. **Machine Quality Type** significantly affects failure rates (L > M > H)
4. Multiple failure modes can be predicted with high confidence
5. The models are production-ready with minimal false alarms

## Files Structure

```
├── Predective_maint.ipynb          # Main analysis notebook
├── dataset/
│   └── ai4i2020.csv               # Raw dataset
├── xgboost_pipeline_model.pkl      # Saved XGBoost model
├── README.md                       # This file
└── .gitignore                     # Git ignore file
```

## Technologies Used

- **Python**: Main programming language
- **Pandas & NumPy**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms and preprocessing
- **XGBoost**: Advanced gradient boosting
- **Imbalanced-learn**: SMOTENC for handling class imbalance
- **Hyperopt**: Bayesian optimization for hyperparameter tuning
- **Matplotlib & Seaborn**: Data visualization
- **Jupyter Notebook**: Interactive development environment

## Usage

1. Clone the repository
2. Install required dependencies: `pip install pandas numpy scikit-learn xgboost imbalanced-learn hyperopt matplotlib seaborn jupyter`
3. Open `Predective_maint.ipynb` in Jupyter Notebook
4. Run the cells sequentially to reproduce the analysis

## Model Deployment

The trained XGBoost model is saved as `xgboost_pipeline_model.pkl` and can be loaded for predictions:

```python
import joblib
model = joblib.load('xgboost_pipeline_model.pkl')
predictions = model.predict(new_sensor_data)
```

## Results Summary

This predictive maintenance solution successfully:
- Identifies 99%+ of equipment failures before they occur
- Minimizes false alarms (high precision)
- Provides actionable insights for maintenance scheduling
- Reduces unplanned downtime and maintenance costs

The model is ready for production deployment in industrial IoT environments.