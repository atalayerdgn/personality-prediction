# Personality Prediction using XGBoost

This project implements a machine learning model to predict personality types using XGBoost algorithm. The model analyzes various features to predict whether a person is extroverted (E) or introverted (I).

## Project Structure

- `personality_prediction_xgboost.py`: Main implementation file containing the XGBoost model for personality prediction
- `xgboost_feature_importance.py`: Implementation of feature importance analysis using XGBoost
- `random_forest_feature_importance.py`: Alternative implementation using Random Forest for feature importance
- `max_frequency_difference.py`: Implementation of a string frequency analysis algorithm
- `lexicographical_order.py`: Implementation of a lexicographical ordering algorithm

## Features

- Personality type prediction (E/I)
- Feature importance analysis
- Model evaluation with cross-validation
- Learning curve analysis
- Performance metrics visualization

## Requirements

```bash
pip install numpy pandas xgboost scikit-learn matplotlib seaborn
```

## Usage

1. Prepare your data:
```python
X = df.drop('Personality 1:E 0:I', axis=1)
y = df['Personality 1:E 0:I']
```

2. Train and evaluate the model:
```python
# Train the model
model, X_train, X_test, y_train, y_test = train_personality_predictor(X, y)

# Evaluate the model
evaluate_model(model, X_train, X_test, y_train, y_test)
```

3. Make predictions:
```python
predictions, probabilities = predict_personality(model, new_data)
```

## Model Performance

The current implementation achieves:
- Average CV Score: 0.9928 (±0.0037)
- High consistency across cross-validation folds
- No signs of overfitting

## Features

1. **Model Training**
   - XGBoost implementation
   - Cross-validation
   - Learning curve analysis

2. **Evaluation Metrics**
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - Confusion Matrix

3. **Visualization**
   - Feature importance plots
   - Learning curves
   - Confusion matrix heatmap
