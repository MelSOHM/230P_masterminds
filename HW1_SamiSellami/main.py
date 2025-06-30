import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import SelectKBest, f_regression, RFE
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import joblib
import warnings
warnings.filterwarnings('ignore')

class FinancialReturnPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.feature_importance_results = None
        self.feature_selector = None
        self.k_features = None
        
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
            # Ensure all possible categories are present
            macro1_categories = ['Expansion', 'Contraction', 'Recovery', 'Peak', 'Trough']
            df['macro1'] = pd.Categorical(df['macro1'], categories=macro1_categories)
            macro1_dummies = pd.get_dummies(df['macro1'], prefix='macro1')
            df = pd.concat([df, macro1_dummies], axis=1)
            engineered_features.extend(macro1_dummies.columns.tolist())
        
        # 2. Create lagged features (1-day, 2-day, and 3-day lags)
        lag_features = ['macro2', 'price', 'firm1', 'firm2', 'firm3']
        for feature in lag_features:
            if feature in df.columns:
                # Create lags within each firm
                for lag in [1, 2, 3]:
                    df[f'{feature}_lag{lag}'] = df.groupby('firm_id')[feature].shift(lag)
                    engineered_features.append(f'{feature}_lag{lag}')
        
        # 3. Create interaction terms
        continuous_features = ['macro2', 'firm1', 'firm2', 'firm3']
        for i, feat1 in enumerate(continuous_features):
            for feat2 in continuous_features[i+1:]:
                if feat1 in df.columns and feat2 in df.columns:
                    df[f'{feat1}_{feat2}_interaction'] = df[feat1] * df[feat2]
                    engineered_features.append(f'{feat1}_{feat2}_interaction')
        
        # 4. Enhanced price-based features
        if 'price' in df.columns:
            # Log price
            df['log_price'] = np.log(df['price'] + 1e-8)
            engineered_features.append('log_price')
            
            # Price momentum (percentage change)
            df['price_momentum'] = df.groupby('firm_id')['price'].pct_change()
            engineered_features.append('price_momentum')
            
            # Price acceleration (change in momentum)
            df['price_acceleration'] = df.groupby('firm_id')['price_momentum'].diff()
            engineered_features.append('price_acceleration')
            
            # Multiple rolling windows for price
            for window in [3, 5, 7, 10]:
                # Moving averages
                df[f'price_ma{window}'] = df.groupby('firm_id')['price'].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
                # Volatility (standard deviation)
                df[f'price_volatility{window}'] = df.groupby('firm_id')['price'].transform(
                    lambda x: x.rolling(window, min_periods=1).std()
                )
                # Price relative to moving average
                df[f'price_ma{window}_ratio'] = df['price'] / (df[f'price_ma{window}'] + 1e-8)
                engineered_features.extend([f'price_ma{window}', f'price_volatility{window}', f'price_ma{window}_ratio'])
            
            # Bollinger Bands (price relative to MA ± 2*std)
            df['price_bb_upper'] = df['price_ma5'] + 2 * df['price_volatility5']
            df['price_bb_lower'] = df['price_ma5'] - 2 * df['price_volatility5']
            df['price_bb_position'] = (df['price'] - df['price_bb_lower']) / (df['price_bb_upper'] - df['price_bb_lower'] + 1e-8)
            engineered_features.extend(['price_bb_upper', 'price_bb_lower', 'price_bb_position'])
        
        # 5. Enhanced firm-specific features
        firm_features = ['firm1', 'firm2', 'firm3']
        for feature in firm_features:
            if feature in df.columns:
                # Multiple rolling windows
                for window in [3, 5, 7]:
                    # Moving averages
                    df[f'{feature}_ma{window}'] = df.groupby('firm_id')[feature].transform(
                        lambda x: x.rolling(window, min_periods=1).mean()
                    )
                    # Volatility
                    df[f'{feature}_volatility{window}'] = df.groupby('firm_id')[feature].transform(
                        lambda x: x.rolling(window, min_periods=1).std()
                    )
                    # Momentum
                    df[f'{feature}_momentum{window}'] = df.groupby('firm_id')[feature].transform(
                        lambda x: x.pct_change(window)
                    )
                    engineered_features.extend([f'{feature}_ma{window}', f'{feature}_volatility{window}', f'{feature}_momentum{window}'])
                
                # Z-score (standardized relative to rolling mean)
                df[f'{feature}_zscore5'] = (df[feature] - df[f'{feature}_ma5']) / (df[f'{feature}_volatility5'] + 1e-8)
                engineered_features.append(f'{feature}_zscore5')
        
        # 6. Macro features enhancement
        if 'macro2' in df.columns:
            # Multiple rolling windows for macro2
            for window in [3, 5, 7]:
                df[f'macro2_ma{window}'] = df.groupby('firm_id')['macro2'].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
                df[f'macro2_volatility{window}'] = df.groupby('firm_id')['macro2'].transform(
                    lambda x: x.rolling(window, min_periods=1).std()
                )
                engineered_features.extend([f'macro2_ma{window}', f'macro2_volatility{window}'])
            
            # Macro2 momentum
            df['macro2_momentum'] = df.groupby('firm_id')['macro2'].pct_change()
            engineered_features.append('macro2_momentum')
        
        # 7. Polynomial features using sklearn's PolynomialFeatures
        poly_features = ['macro2', 'firm1', 'firm2', 'firm3']
        poly_features_exist = [f for f in poly_features if f in df.columns]
        
        if poly_features_exist:
            # Create polynomial features (degree=2 includes squared and interaction terms)
            poly = PolynomialFeatures(degree=2, include_bias=False)
            poly_data = poly.fit_transform(df[poly_features_exist])
            
            # Get feature names for polynomial features
            poly_feature_names = poly.get_feature_names_out(poly_features_exist)
            
            # Add polynomial features to dataframe
            for i, feature_name in enumerate(poly_feature_names):
                if feature_name not in df.columns:  # Avoid duplicates
                    df[feature_name] = poly_data[:, i]
                    engineered_features.append(feature_name)
        
        # 8. Enhanced cross-sectional features (firm rankings within each date)
        ranking_features = ['firm1', 'firm2', 'firm3', 'price']
        for feature in ranking_features:
            if feature in df.columns:
                # Percentile rank
                df[f'{feature}_rank'] = df.groupby('date')[feature].rank(pct=True)
                # Z-score across firms on each date
                df[f'{feature}_cross_zscore'] = df.groupby('date')[feature].transform(
                    lambda x: (x - x.mean()) / (x.std() + 1e-8)
                )
                engineered_features.extend([f'{feature}_rank', f'{feature}_cross_zscore'])
        
        # 9. Time-based features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        engineered_features.extend(['day_of_week', 'month', 'quarter'])
        
        # 10. Statistical features
        for feature in ['firm1', 'firm2', 'firm3']:
            if feature in df.columns:
                # Rolling skewness and kurtosis
                df[f'{feature}_skew5'] = df.groupby('firm_id')[feature].transform(
                    lambda x: x.rolling(5, min_periods=1).skew()
                )
                df[f'{feature}_kurt5'] = df.groupby('firm_id')[feature].transform(
                    lambda x: x.rolling(5, min_periods=1).kurt()
                )
                engineered_features.extend([f'{feature}_skew5', f'{feature}_kurt5'])
        
        # 11. Market regime features (using macro2 as proxy)
        if 'macro2' in df.columns:
            # Market regime based on macro2 volatility
            df['market_regime'] = df.groupby('firm_id')['macro2'].transform(
                lambda x: pd.cut(x.rolling(10, min_periods=1).std(), 
                               bins=3, labels=['low_vol', 'med_vol', 'high_vol'])
            )
            # One-hot encode market regime
            regime_dummies = pd.get_dummies(df['market_regime'], prefix='market_regime')
            df = pd.concat([df, regime_dummies], axis=1)
            engineered_features.extend(regime_dummies.columns.tolist())
        
        # Combine original continuous features with engineered features
        continuous_originals = ['macro2', 'price', 'firm1', 'firm2', 'firm3']
        all_features = [f for f in continuous_originals if f in df.columns] + engineered_features
        
        return df, all_features
    
    def prepare_data(self, df, is_training=True, feature_selection_method='kbest'):
        """
        Prepare data for modeling with feature selection (SelectKBest or RFE)
        """
        # Feature engineering
        df_engineered, feature_names = self.engineer_features(df)
        
        if is_training:
            self.feature_names = feature_names
        
        # Select features
        X = df_engineered[self.feature_names].copy()
        
        # Handle missing values (forward fill within each firm)
        X = X.groupby(df_engineered['firm_id']).fillna(method='ffill').fillna(0)
        
        # Feature selection (only for training)
        if is_training:
            if feature_selection_method == 'kbest':
                # Select top k features based on F-statistic
                self.feature_selector = SelectKBest(score_func=f_regression, k=self.k_features)
                X_selected = self.feature_selector.fit_transform(X, df_engineered['ret'])
                selected_feature_names = [self.feature_names[i] for i in self.feature_selector.get_support(indices=True)]
                self.feature_names = selected_feature_names
                X = pd.DataFrame(X_selected, columns=selected_feature_names, index=X.index)
            elif feature_selection_method == 'rfe':
                # Use RFE with ElasticNet
                estimator = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42, max_iter=2000)
                self.feature_selector = RFE(estimator, n_features_to_select=self.k_features, step=0.1)
                X_selected = self.feature_selector.fit_transform(X, df_engineered['ret'])
                selected_feature_names = [self.feature_names[i] for i, s in enumerate(self.feature_selector.support_) if s]
                self.feature_names = selected_feature_names
                X = pd.DataFrame(X_selected, columns=selected_feature_names, index=X.index)
        else:
            # For validation/prediction, ensure we have the same features as training
            if self.feature_names is None:
                raise ValueError("Model must be trained first before preparing validation data.")
            
            # Add missing features with zeros (for categorical one-hot encoded features)
            missing_features = set(self.feature_names) - set(df_engineered.columns)
            for feature in missing_features:
                df_engineered[feature] = 0
            
            # Ensure we have exactly the same columns as training, in the same order
            for feature in self.feature_names:
                if feature not in df_engineered.columns:
                    df_engineered[feature] = 0
            
            X = df_engineered[self.feature_names].copy()
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
    
    def time_series_split(self, df, split_ratio=0.8, validation_strategy='fixed'):
        """
        Create time-based train-validation split with multiple strategies
        """
        unique_dates = sorted(df['date'].unique())
        
        if validation_strategy == 'fixed':
            # Original fixed split
            cutoff_idx = int(len(unique_dates) * split_ratio)
            cutoff_date = unique_dates[cutoff_idx]
            
            train_data = df[df['date'] <= cutoff_date].copy()
            val_data = df[df['date'] > cutoff_date].copy()
            
        elif validation_strategy == 'longer_validation':
            # Use longer validation period (70/30 split)
            cutoff_idx = int(len(unique_dates) * 0.7)
            cutoff_date = unique_dates[cutoff_idx]
            
            train_data = df[df['date'] <= cutoff_date].copy()
            val_data = df[df['date'] > cutoff_date].copy()
            
        elif validation_strategy == 'middle_validation':
            # Use middle period for validation (avoid potential regime changes at start/end)
            total_days = len(unique_dates)
            train_start = int(total_days * 0.2)  # Skip first 20%
            train_end = int(total_days * 0.7)    # End at 70%
            val_start = train_end
            val_end = int(total_days * 0.9)      # End at 90%
            
            train_dates = unique_dates[train_start:train_end]
            val_dates = unique_dates[val_start:val_end]
            
            train_data = df[df['date'].isin(train_dates)].copy()
            val_data = df[df['date'].isin(val_dates)].copy()
            
        elif validation_strategy == 'recent_validation':
            # Use most recent data for validation (more realistic for production)
            cutoff_idx = int(len(unique_dates) * 0.75)
            cutoff_date = unique_dates[cutoff_idx]
            
            train_data = df[df['date'] <= cutoff_date].copy()
            val_data = df[df['date'] > cutoff_date].copy()
        
        print(f"Validation Strategy: {validation_strategy}")
        print(f"Training period: {train_data['date'].min()} to {train_data['date'].max()}")
        print(f"Validation period: {val_data['date'].min()} to {val_data['date'].max()}")
        print(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")
        
        return train_data, val_data
    
    def tune_hyperparameters(self, X_train, y_train, cv_splits=5):
        """
        Tune ElasticNet hyperparameters using time series cross-validation
        """
        # Create time series splits for cross-validation
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        
        # Define parameter grid
        param_grid = {
            'alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0],
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
        }
        
        # Create ElasticNet model
        elastic_net = ElasticNet(random_state=42, max_iter=2000)
        
        # Grid search with time series CV
        grid_search = GridSearchCV(
            elastic_net, 
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
    
    def fit(self, df, split_ratio=0.8, validation_strategy='fixed', feature_selection_method='kbest'):
        """
        Train the complete model pipeline
        """
        # Time-based split with improved validation strategy
        train_data, val_data = self.time_series_split(df, split_ratio, validation_strategy)
        
        # Prepare training data
        X_train, y_train, train_meta = self.prepare_data(train_data, is_training=True, feature_selection_method=feature_selection_method)
        
        # Prepare validation data
        X_val, y_val, val_meta = self.prepare_data(val_data, is_training=False, feature_selection_method=feature_selection_method)
        
        print(f"Training features shape: {X_train.shape}")
        print(f"Validation features shape: {X_val.shape}")
        print(f"Selected features: {self.feature_names}")
        
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
    
    def calculate_feature_importance(self, n_repeats=10, random_state=42):
        """
        Calculate permutation-based feature importance with statistical significance
        """
        if self.model is None or not hasattr(self, 'X_val'):
            raise ValueError("Model must be fitted first and validation data must be available.")
        
        # Calculate permutation importance
        perm_importance = permutation_importance(
            self.model, self.X_val, self.y_val,
            n_repeats=n_repeats,
            random_state=random_state,
            scoring='r2'
        )
        
        # Prepare results
        results_list = []
        
        if self.feature_names is None:
            raise ValueError("Feature names not available. Train the model first.")
        
        for i, feature in enumerate(self.feature_names):
            importance_scores = perm_importance.importances[i]
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
        plt.xlabel('Permutation Importance (R² decrease)')
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
        
        print("=== FEATURE IMPORTANCE INTERPRETATION ===\n")
        
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
    
    def save_model(self, filepath='model.pkl'):
        """
        Save the trained model
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='model.pkl'):
        """
        Load a trained model
        """
        self.model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")

def main():
    """
    Main execution function
    """
    print("=== Financial Return Prediction with Elastic Net ===\n")
    
    # Load data
    print("Loading training data...")
    df = pd.read_parquet('training_data.parquet')
    
    # Basic data exploration
    print(f"Dataset shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Number of firms: {df['firm_id'].nunique()}")
    print(f"Features: {df.columns.tolist()}")
    print(f"Missing values:\n{df.isnull().sum()}\n")
    
    # Grid search over number of features and feature selection method
    best_r2 = -np.inf
    best_k = None
    best_results = None
    best_predictor = None
    best_importance_results = None
    best_method = None
    
    for method in ['kbest', 'rfe']:
        for k in [20, 30, 40, 50, 60]:
            print(f"\n--- Trying {method} with k={k} features ---")
            predictor = FinancialReturnPredictor()
            predictor.feature_selector = None  # Reset
            predictor.k_features = k  # Set k for this run
            results = predictor.fit(df, split_ratio=0.8, validation_strategy='fixed', feature_selection_method=method)
            importance_results = predictor.calculate_feature_importance(n_repeats=10)
            print(f"{method} k={k}: Validation R²={results['val_r2']:.4f}, RMSE={results['val_rmse']:.4f}, DirAcc={results['directional_accuracy']:.4f}")
            if results['val_r2'] > best_r2:
                best_r2 = results['val_r2']
                best_k = k
                best_results = results
                best_predictor = predictor
                best_importance_results = importance_results
                best_method = method
    
    print(f"\n=== BEST RESULT ===")
    print(f"Best method: {best_method}")
    print(f"Best k: {best_k}")
    print(f"Best Validation R²: {best_r2:.4f}")
    print(f"Other metrics: {best_results}")
    
    # Save results for best k
    best_importance_results.to_csv('feature_importance_results.csv', index=False)
    print("Feature importance results saved to 'feature_importance_results.csv'")
    best_predictor.plot_feature_importance(top_n=20)
    significant_features = best_predictor.interpret_results()
    best_predictor.save_model('model.pkl')
    
    print(f"\n=== SUMMARY ===")
    print(f"Best hyperparameters: {best_results['best_params']}")
    print(f"Validation R²: {best_results['val_r2']:.4f}")
    print(f"Validation RMSE: {best_results['val_rmse']:.4f}")
    print(f"Directional Accuracy: {best_results['directional_accuracy']:.4f}")
    print(f"Significant features: {len(significant_features)}/{len(best_importance_results)}")
    
    return best_predictor, best_results, best_importance_results

if __name__ == "__main__":
    predictor, results, importance_results = main()