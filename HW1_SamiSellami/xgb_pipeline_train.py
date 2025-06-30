import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import joblib
from feature_engineering import FeatureEngineeringTransformer

# Load data
print('Loading data...')
df = pd.read_parquet('training_data.parquet')
X = df.drop(columns=['ret'])
y = df['ret']

# Build pipeline
pipeline = Pipeline([
    ('features', FeatureEngineeringTransformer()),
    ('scaler', StandardScaler()),
    ('model', XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, objective='reg:squarederror'))
])

print('Training pipeline...')
pipeline.fit(X, y)

# Save pipeline
joblib.dump(pipeline, 'model_xgboost.pkl')
print('Pipeline saved to model_xgboost.pkl')

# Test: load and predict as in the professor's code
print('Testing saved model...')
your_model = joblib.load('model_xgboost.pkl')
df_test = pd.read_parquet('training_data.parquet')
preds = your_model.predict(df_test.drop(columns=['ret']))
print('Predictions shape:', preds.shape)
print('First 5 predictions:', preds[:5]) 