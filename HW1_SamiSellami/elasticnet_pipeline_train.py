import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
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
    ('model', ElasticNet(alpha=0.001, l1_ratio=0.1, max_iter=2000, random_state=42))
])

print('Training pipeline...')
pipeline.fit(X, y)

# Save pipeline
joblib.dump(pipeline, 'model_elasticnet.pkl')
print('Pipeline saved to model_elasticnet.pkl')

# Test: load and predict as in the professor's code
print('Testing saved model...')
your_model = joblib.load('model_elasticnet.pkl')
df_test = pd.read_parquet('training_data.parquet')
preds = your_model.predict(df_test)
print('Predictions shape:', preds.shape)
print('First 5 predictions:', preds[:5]) 