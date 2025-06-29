import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import joblib
import warnings
warnings.filterwarnings('ignore')

class FinancialReturnPredictorXGBoost:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.feature_importance_results = None
        
    def engineer_features(self, df):
        """
        Comprehensive feature engineering for financial data
        """
        # Sort by firm_id and date to ensure proper ordering
        df = df.sort_values(['firm_id', 'date']).copy()
        
        # Store original feature names
        original_features = ['macro1', 'macro2', 'price', 'firm1', 'firm2', 'firm3']
        engineered_features = []
        
        # 1. Handle categorical variable (macro1)
        if 'macro1' in df.columns:
            # One-hot encode macro1
            macro1_dummies = pd.get_dummies(df['macro1'], prefix='macro1')
            df = pd.concat([df, macro1_dummies], axis=1)
            engineered_features.extend(macro1_dummies.columns.tolist())
        
        # 2. Create lagged features (1-day and 2-day lags)
        lag_features = ['macro2', 'price', 'firm1', 'firm2', 'firm3']
        for feature in lag_features:
            if feature in df.columns:
                # Create lags within each firm
                df[f'{feature}_lag1'] = df.groupby('firm_id')[feature].shift(1)
                df[f'{feature}_lag2'] = df.groupby('firm_id')[feature].shift(2)
                engineered_features.extend([f'{feature}_lag1', f'{feature}_lag2'])
        
        # 3. Create interaction terms
        continuous_features = ['macro2', 'firm1', 'firm2', 'firm3']
        for i, feat1 in enumerate(continuous_features):
            for feat2 in continuous_features[i+1:]:
                if feat1 in df.columns and feat2 in df.columns:
                    df[f'{feat1}_{feat2}_interaction'] = df[feat1] * df[feat2]
                    engineered_features.append(f'{feat1}_{feat2}_interaction')
        
        # 4. Price-based features
        if 'price' in df.columns:
            # Log price
            df['log_price'] = np.log(df['price'] + 1e-8)  # Add small constant to avoid log(0)
            engineered_features.append('log_price')
            
            # Price momentum (difference from lag)
            df['price_momentum'] = df.groupby('firm_id')['price'].pct_change()
            engineered_features.append('price_momentum')
            
            # Rolling statistics (5-day window)
            df['price_ma5'] = df.groupby('firm_id')['price'].transform(lambda x: x.rolling(5, min_periods=1).mean())
            df['price_volatility5'] = df.groupby('firm_id')['price'].transform(lambda x: x.rolling(5, min_periods=1).std())
            engineered_features.extend(['price_ma5', 'price_volatility5'])
        
        # 5. Polynomial features (squared terms)
        poly_features = ['macro2', 'firm1', 'firm2', 'firm3']
        for feature in poly_features:
            if feature in df.columns:
                df[f'{feature}_squared'] = df[feature] ** 2
                engineered_features.append(f'{feature}_squared')
        
        # 6. Cross-sectional features (firm rankings within each date)
        ranking_features = ['firm1', 'firm2', 'firm3', 'price']
        for feature in ranking_features:
            if feature in df.columns:
                df[f'{feature}_rank'] = df.groupby('date')[feature].rank(pct=True)
                engineered_features.append(f'{feature}_rank')
        
        # Combine original continuous features with engineered features
        continuous_originals = ['macro2', 'price', 'firm1', 'firm2', 'firm3']
        all_features = [f for f in continuous_originals if f in df.columns] + engineered_features
        
        return df, all_features
    
    def prepare_data(self, df, is_training=True):
        """
        Prepare data for modeling
        """
        # Feature engineering
        df_engineered, feature_names = self.engineer_features(df)
        
        if is_training:
            self.feature_names = feature_names
        else:
            # For validation/prediction, ensure we have the same features as training
            if self.feature_names is None:
                raise ValueError("Model must be trained first before preparing validation data.")
            
            # Add missing features with zeros
            missing_features = set(self.feature_names) - set(df_engineered.columns)
            for feature in missing_features:
                df_engineered[feature] = 0
            
            # Ensure we have exactly the same columns as training, in the same order
            for feature in self.feature_names:
                if feature not in df_engineered.columns:
                    df_engineered[feature] = 0
        
        # Select features in the correct order
        X = df_engineered[self.feature_names].copy()
        
        # Handle missing values (forward fill within each firm)
        X = X.groupby(df_engineered['firm_id']).fillna(method='ffill').fillna(0)
        
        # Scale features
        if is_training:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names, index=X.index)
        
        # Return target if available
        if 'ret' in df_engineered.columns:
            y = df_engineered['ret'].values
            return X_scaled, y, df_engineered[['date', 'firm_id']].copy()
        else:
            return X_scaled, None, df_engineered[['date', 'firm_id']].copy()
    
    def time_series_split(self, df, split_ratio=0.8):
        """
        Create time-based train-validation split
        """
        # Find the cutoff date
        unique_dates = sorted(df['date'].unique())
        cutoff_idx = int(len(unique_dates) * split_ratio)
        cutoff_date = unique_dates[cutoff_idx]
        
        train_data = df[df['date'] <= cutoff_date].copy()
        val_data = df[df['date'] > cutoff_date].copy()
        
        print(f"Training period: {train_data['date'].min()} to {train_data['date'].max()}")
        print(f"Validation period: {val_data['date'].min()} to {val_data['date'].max()}")
        print(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")
        
        return train_data, val_data
    
    def tune_hyperparameters(self, X_train, y_train, cv_splits=5):
        """
        Tune XGBoost hyperparameters using time series cross-validation
        """
        # Create time series splits for cross-validation
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        
        # Define parameter grid for XGBoost (reduced for faster execution)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9]
        }
        
        # Create XGBoost model
        xgb_model = xgb.XGBRegressor(
            random_state=42,
            objective='reg:squarederror',
            eval_metric='rmse'
        )
        
        # Grid search with time series CV
        grid_search = GridSearchCV(
            xgb_model, 
            param_grid, 
            cv=tscv, 
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit grid search
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_, grid_search.best_params_
    
    def fit(self, df, split_ratio=0.8):
        """
        Train the complete model pipeline
        """
        # Time-based split
        train_data, val_data = self.time_series_split(df, split_ratio)
        
        # Prepare training data
        X_train, y_train, train_meta = self.prepare_data(train_data, is_training=True)
        
        # Prepare validation data
        X_val, y_val, val_meta = self.prepare_data(val_data, is_training=False)
        
        print(f"Training features shape: {X_train.shape}")
        print(f"Validation features shape: {X_val.shape}")
        
        # Hyperparameter tuning
        self.model, best_params = self.tune_hyperparameters(X_train, y_train)
        
        # Evaluate on validation set
        y_val_pred = self.model.predict(X_val)
        val_r2 = r2_score(y_val, y_val_pred)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        
        print(f"\nValidation Performance:")
        print(f"R² Score: {val_r2:.4f}")
        print(f"RMSE: {val_rmse:.4f}")
        
        # Directional accuracy
        actual_direction = np.sign(y_val)
        predicted_direction = np.sign(y_val_pred)
        directional_accuracy = np.mean(actual_direction == predicted_direction)
        print(f"Directional Accuracy: {directional_accuracy:.4f}")
        
        # Store validation data for feature importance analysis
        self.X_val = X_val
        self.y_val = y_val
        
        return {
            'best_params': best_params,
            'val_r2': val_r2,
            'val_rmse': val_rmse,
            'directional_accuracy': directional_accuracy
        }
    
    def predict(self, df):
        """
        Make predictions on new data
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        X, _, _ = self.prepare_data(df, is_training=False)
        return self.model.predict(X)
    
    def calculate_feature_importance(self, n_bootstrap=100, random_state=123):
        """
        Calculate XGBoost feature importance with bootstrapping for p-values
        """
        if self.model is None or not hasattr(self, 'X_val'):
            raise ValueError("Model must be fitted first and validation data must be available.")
        
        if self.feature_names is None:
            raise ValueError("Feature names not available. Train the model first.")
        
        # Get XGBoost's built-in feature importance (number of splits)
        xgb_importance = self.model.feature_importances_
        
        # Bootstrap to estimate p-values
        np.random.seed(random_state)
        bootstrap_importances = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample indices
            bootstrap_indices = np.random.choice(
                len(self.X_val), 
                size=len(self.X_val), 
                replace=True
            )
            
            # Create bootstrap sample
            X_bootstrap = self.X_val.iloc[bootstrap_indices]
            y_bootstrap = self.y_val[bootstrap_indices]
            
            # Train a new XGBoost model on bootstrap sample
            bootstrap_model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=random_state,
                objective='reg:squarederror'
            )
            bootstrap_model.fit(X_bootstrap, y_bootstrap)
            
            # Get feature importance from bootstrap model
            bootstrap_importance = bootstrap_model.feature_importances_
            bootstrap_importances.append(bootstrap_importance)
        
        # Convert to numpy array
        bootstrap_importances = np.array(bootstrap_importances)
        
        # Prepare results
        results_list = []
        
        for i, feature in enumerate(self.feature_names):
            importance_scores = bootstrap_importances[:, i]
            mean_importance = importance_scores.mean()
            std_importance = importance_scores.std()
            
            # Statistical significance test (one-sample t-test against 0)
            t_stat, p_value = stats.ttest_1samp(importance_scores, 0)
            
            results_list.append({
                'feature': feature,
                'importance_mean': mean_importance,
                'importance_std': std_importance,
                't_statistic': t_stat,
                'p_value': p_value,
                'is_significant': p_value < 0.05
            })
        
        # Convert to DataFrame and sort by importance
        self.feature_importance_results = pd.DataFrame(results_list)
        self.feature_importance_results = self.feature_importance_results.sort_values(
            'importance_mean', ascending=False
        ).reset_index(drop=True)
        
        return self.feature_importance_results
    
    def plot_feature_importance(self, top_n=20):
        """
        Visualize feature importance with significance indicators
        """
        if self.feature_importance_results is None:
            raise ValueError("Feature importance has not been calculated yet.")
        
        # Select top N features
        top_features = self.feature_importance_results.head(top_n).copy()
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Color based on significance
        colors = ['red' if sig else 'gray' for sig in top_features['is_significant']]
        
        bars = plt.barh(range(len(top_features)), top_features['importance_mean'], 
                       color=colors, alpha=0.7)
        
        # Add error bars
        plt.errorbar(top_features['importance_mean'], range(len(top_features)),
                    xerr=top_features['importance_std'], fmt='none', color='black', alpha=0.5)
        
        # Customize plot
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('XGBoost Feature Importance (Number of Splits)')
        plt.title(f'Top {top_n} Feature Importance (Red = Significant at p<0.05)')
        plt.grid(axis='x', alpha=0.3)
        
        # Add significance legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='red', alpha=0.7, label='Significant (p<0.05)'),
                          Patch(facecolor='gray', alpha=0.7, label='Not Significant')]
        plt.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        plt.show()
        
        return plt.gcf()
    
    def interpret_results(self):
        """
        Provide interpretation of feature importance results
        """
        if self.feature_importance_results is None:
            raise ValueError("Feature importance has not been calculated yet.")
        
        significant_features = self.feature_importance_results[
            self.feature_importance_results['is_significant']
        ]
        
        print("=== XGBOOST FEATURE IMPORTANCE INTERPRETATION ===\n")
        
        print(f"Total features analyzed: {len(self.feature_importance_results)}")
        print(f"Statistically significant features: {len(significant_features)}")
        print(f"Significance rate: {len(significant_features)/len(self.feature_importance_results)*100:.1f}%\n")
        
        print("Top 10 Most Important Features:")
        print("-" * 50)
        for idx, row in self.feature_importance_results.head(10).iterrows():
            sig_marker = "***" if row['is_significant'] else ""
            print(f"{idx+1:2d}. {row['feature']:<25} | "
                  f"Importance: {row['importance_mean']:>7.4f} ± {row['importance_std']:>6.4f} | "
                  f"p-value: {row['p_value']:>7.4f} {sig_marker}")
        
        return significant_features
    
    def save_model(self, filepath='model_xgboost.pkl'):
        """
        Save the trained model
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'label_encoders': self.label_encoders
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='model_xgboost.pkl'):
        """
        Load a trained model
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.label_encoders = model_data['label_encoders']
        
        print(f"Model loaded from {filepath}")

def main():
    """
    Main execution function
    """
    print("=== Financial Return Prediction with XGBoost ===\n")
    
    # Load data
    print("Loading training data...")
    df = pd.read_parquet('training_data.parquet')
    
    # Basic data exploration
    print(f"Dataset shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Number of firms: {df['firm_id'].nunique()}")
    print(f"Features: {df.columns.tolist()}")
    print(f"Missing values:\n{df.isnull().sum()}\n")
    
    # Initialize and train model
    predictor = FinancialReturnPredictorXGBoost()
    
    print("Training model with hyperparameter tuning...")
    results = predictor.fit(df, split_ratio=0.8)
    
    # Feature importance analysis
    print("\nCalculating feature importance...")
    importance_results = predictor.calculate_feature_importance(n_bootstrap=100)
    
    # Save results
    importance_results.to_csv('feature_importance_results_xgboost.csv', index=False)
    print("Feature importance results saved to 'feature_importance_results_xgboost.csv'")
    
    # Visualize and interpret results
    predictor.plot_feature_importance(top_n=20)
    significant_features = predictor.interpret_results()
    
    # Save model
    predictor.save_model('model_xgboost.pkl')
    
    print(f"\n=== SUMMARY ===")
    print(f"Best hyperparameters: {results['best_params']}")
    print(f"Validation R²: {results['val_r2']:.4f}")
    print(f"Validation RMSE: {results['val_rmse']:.4f}")
    print(f"Directional Accuracy: {results['directional_accuracy']:.4f}")
    print(f"Significant features: {len(significant_features)}/{len(importance_results)}")
    
    return predictor, results, importance_results

if __name__ == "__main__":
    predictor, results, importance_results = main() 