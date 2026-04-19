"""
Model Training and Evaluation Utilities

This module provides reusable functions for:
- Data preparation and feature engineering
- Model training with hyperparameter tuning
- Model evaluation and comparison
- Experiment tracking and results management

Methodology Notes:
- All functions follow scikit-learn conventions
- Three feature sets enable progressive complexity analysis
- Comprehensive metrics ensure robust evaluation
- Experiment tracking maintains reproducibility
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, List, Any, Optional

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, accuracy_score, roc_auc_score, 
    confusion_matrix, classification_report, roc_curve, auc
)
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


class ExperimentTracker:
    """
    Tracks model training experiments with timestamps and results.
    
    Purpose:
    - Save experiment metadata and configurations
    - Store model predictions and metrics
    - Enable reproducibility with exact hyperparameters
    - Maintain history for comparison
    
    Attributes:
        experiment_dir (str): Directory for storing experiment results
        experiment_name (str): Unique identifier for experiment
        timestamp (str): ISO format timestamp of experiment start
    """
    
    def __init__(self, base_dir: str = "experiments"):
        """
        Initialize experiment tracker.
        
        Args:
            base_dir: Base directory for storing experiments
        """
        self.base_dir = base_dir
        Path(base_dir).mkdir(exist_ok=True)
        
    def create_experiment(self, name: str) -> str:
        """
        Create new experiment directory with timestamp.
        
        Args:
            name: Descriptive name for the experiment
            
        Returns:
            str: Path to experiment directory
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = f"{name}_{timestamp}"
        self.experiment_dir = os.path.join(self.base_dir, self.experiment_name)
        self.timestamp = datetime.now().isoformat()
        
        Path(self.experiment_dir).mkdir(parents=True, exist_ok=True)
        return self.experiment_dir
    
    def save_config(self, config: Dict[str, Any]) -> None:
        """
        Save experiment configuration.
        
        Args:
            config: Configuration dictionary with hyperparameters
        """
        config_path = os.path.join(self.experiment_dir, "config.json")
        config['timestamp'] = self.timestamp
        config['experiment_name'] = self.experiment_name
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
    
    def save_results(self, results: Dict[str, Any]) -> None:
        """
        Save experiment results and metrics.
        
        Args:
            results: Dictionary containing model metrics and predictions
        """
        results_path = os.path.join(self.experiment_dir, "results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    def save_model(self, model: Any, name: str) -> None:
        """
        Save trained model to pickle file.
        
        Args:
            model: Trained model object
            name: Model identifier
        """
        model_path = os.path.join(self.experiment_dir, f"{name}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    
    def get_experiment_summary(self) -> Dict[str, str]:
        """Get summary of current experiment."""
        return {
            'name': self.experiment_name,
            'directory': self.experiment_dir,
            'timestamp': self.timestamp
        }


class DataPreparation:
    """
    Handles data loading, preprocessing, and feature engineering.
    
    Purpose:
    - Consistent data preprocessing across experiments
    - Feature set management (early, mid, full)
    - Encoding and scaling for different algorithms
    - Train-test split with stratification
    
    Methodology:
    - LabelEncoding: Handles categorical variables (tree-compatible)
    - StandardScaling: Applied only for linear models
    - Stratified splits: Maintains class distribution in small datasets
    """
    
    def __init__(self, data_path: str):
        """
        Initialize data preparation.
        
        Args:
            data_path: Path to CSV data file
        """
        self.data_path = data_path
        self.df = None
        self.label_encoders = {}
        self.scalers = {}
        
    def load_data(self, sep: str = ';') -> pd.DataFrame:
        """
        Load and validate data.
        
        Args:
            sep: CSV separator character
            
        Returns:
            pd.DataFrame: Loaded data
        """
        self.df = pd.read_csv(self.data_path, sep=sep)
        print(f"✓ Data loaded: {self.df.shape}")
        return self.df
    
    def create_feature_sets(
        self, 
        demographic_features: List[str],
        sem1_features: List[str],
        sem2_features: List[str],
        target_col: str = 'Target'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create three feature sets with progressive complexity.
        
        Args:
            demographic_features: Enrollment-time features
            sem1_features: First semester features
            sem2_features: Second semester features
            target_col: Target column name
            
        Returns:
            Tuple of (df_early, df_mid, df_full) DataFrames
            
        Methodology:
        - df_early: Tests demographic predictability (enrollment-time prediction)
        - df_mid: Adds 1st semester data (early intervention point)
        - df_full: Includes 2nd semester data (complete picture)
        """
        # Filter available features
        available = set(self.df.columns)
        demo_avail = [f for f in demographic_features if f in available]
        sem1_avail = [f for f in sem1_features if f in available]
        sem2_avail = [f for f in sem2_features if f in available]
        
        # Create datasets
        df_early = self.df[demo_avail + [target_col]].copy()
        df_mid = self.df[demo_avail + sem1_avail + [target_col]].copy()
        df_full = self.df[demo_avail + sem1_avail + sem2_avail + [target_col]].copy()
        
        print(f"✓ Feature sets created:")
        print(f"  df_early: {df_early.shape}")
        print(f"  df_mid: {df_mid.shape}")
        print(f"  df_full: {df_full.shape}")
        
        return df_early, df_mid, df_full
    
    def encode_and_scale(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        scale: bool = False,
        target_col: str = 'Target'
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Encode categorical variables and optionally scale.
        
        Args:
            df: Input DataFrame
            dataset_name: Name for storing encoders/scalers
            scale: Whether to apply StandardScaling
            target_col: Target column name
            
        Returns:
            Tuple of (X_processed, y_encoded, metadata)
            
        Methodology:
        - LabelEncoding: Preserves tree model efficiency
        - StandardScaling: Improves linear model convergence
        - Separate handling for categorical vs numeric features
        """
        X = df.drop(columns=[target_col]).copy()
        y = df[target_col].copy()
        
        metadata = {'features': X.columns.tolist(), 'target_classes': y.unique().tolist()}
        
        # Encode target
        le_target = LabelEncoder()
        y_encoded = le_target.fit_transform(y)
        metadata['class_mapping'] = dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))
        
        # Encode categorical features
        self.label_encoders[dataset_name] = {}
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[dataset_name][col] = le
        
        X_processed = X.values.astype(float)
        
        # Scale if requested (for linear models)
        if scale:
            scaler = StandardScaler()
            X_processed = scaler.fit_transform(X_processed)
            self.scalers[dataset_name] = scaler
        
        return X_processed, y_encoded, metadata


class ModelTrainer:
    """
    Trains and tunes machine learning models.
    
    Purpose:
    - Consistent model training across algorithms
    - Hyperparameter tuning with cross-validation
    - Nested CV for robust hyperparameter selection
    - Tracks training time and convergence
    
    Methodology:
    - GridSearchCV: Exhaustive search for specified parameters
    - RandomizedSearchCV: Efficient search for large spaces
    - Nested cross-validation: Prevents overfitting in tuning
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize model trainer.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.tuning_history = {}
        
    def train_logistic_regression(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        param_grid: Optional[Dict] = None,
        use_grid_search: bool = True
    ) -> Tuple[LogisticRegression, Dict]:
        """
        Train logistic regression with optional hyperparameter tuning.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            param_grid: Hyperparameter grid to search
            use_grid_search: Whether to perform tuning
            
        Returns:
            Tuple of (trained_model, results_dict)
            
        Methodology:
        - Ridge (L2): Shrinks coefficients, handles multicollinearity
        - Lasso (L1): Can zero out features, automatic selection
        - max_iter: Increased for convergence on complex data
        """
        if param_grid is None:
            param_grid = {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l2', 'l1'],
                'solver': ['saga']
            }
        
        if use_grid_search:
            base_model = LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                multi_class='multinomial'
            )
            gs = GridSearchCV(base_model, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
            gs.fit(X_train, y_train)
            model = gs.best_estimator_
            best_params = gs.best_params_
            best_score = gs.best_score_
        else:
            model = LogisticRegression(
                C=1.0,
                penalty='l2',
                solver='saga',
                max_iter=1000,
                random_state=self.random_state,
                multi_class='multinomial'
            )
            model.fit(X_train, y_train)
            best_params = model.get_params()
            best_score = model.score(X_test, y_test)
        
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted')
        acc = accuracy_score(y_test, y_pred)
        
        results = {
            'model': model,
            'best_params': best_params,
            'best_cv_score': float(best_score),
            'test_f1': float(f1),
            'test_accuracy': float(acc),
            'algorithm': 'Logistic Regression'
        }
        
        return model, results
    
    def train_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        param_grid: Optional[Dict] = None,
        use_random_search: bool = True
    ) -> Tuple[RandomForestClassifier, Dict]:
        """
        Train random forest with optional hyperparameter tuning.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            param_grid: Hyperparameter grid to search
            use_random_search: Whether to use RandomizedSearchCV
            
        Returns:
            Tuple of (trained_model, results_dict)
            
        Methodology:
        - n_estimators: More trees = better but slower
        - max_depth: Controls model complexity
        - min_samples_split: Prevents overfitting
        - RandomizedSearchCV: Efficient for large spaces
        """
        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        
        base_model = RandomForestClassifier(
            random_state=self.random_state,
            n_jobs=-1
        )
        
        if use_random_search:
            rs = RandomizedSearchCV(
                base_model, param_grid, n_iter=20,
                cv=5, scoring='f1_weighted', n_jobs=-1, random_state=self.random_state
            )
            rs.fit(X_train, y_train)
            model = rs.best_estimator_
            best_params = rs.best_params_
            best_score = rs.best_score_
        else:
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            best_params = model.get_params()
            best_score = model.score(X_test, y_test)
        
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted')
        acc = accuracy_score(y_test, y_pred)
        
        results = {
            'model': model,
            'best_params': best_params,
            'best_cv_score': float(best_score),
            'test_f1': float(f1),
            'test_accuracy': float(acc),
            'algorithm': 'Random Forest',
            'feature_importances': model.feature_importances_.tolist()
        }
        
        return model, results
    
    def train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        param_grid: Optional[Dict] = None,
        use_random_search: bool = True
    ) -> Tuple[xgb.XGBClassifier, Dict]:
        """
        Train XGBoost with optional hyperparameter tuning.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            param_grid: Hyperparameter grid to search
            use_random_search: Whether to use RandomizedSearchCV
            
        Returns:
            Tuple of (trained_model, results_dict)
            
        Methodology:
        - learning_rate: Step size for boosting
        - max_depth: Tree complexity control
        - subsample: Row subsampling for robustness
        - colsample_bytree: Column subsampling
        """
        if param_grid is None:
            param_grid = {
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'n_estimators': [100, 200]
            }
        
        base_model = xgb.XGBClassifier(
            random_state=self.random_state,
            n_jobs=-1,
            eval_metric='mlogloss'
        )
        
        if use_random_search:
            rs = RandomizedSearchCV(
                base_model, param_grid, n_iter=20,
                cv=5, scoring='f1_weighted', n_jobs=-1, random_state=self.random_state
            )
            rs.fit(X_train, y_train)
            model = rs.best_estimator_
            best_params = rs.best_params_
            best_score = rs.best_score_
        else:
            model = xgb.XGBClassifier(
                learning_rate=0.1,
                max_depth=5,
                subsample=0.9,
                colsample_bytree=0.9,
                n_estimators=200,
                random_state=self.random_state,
                n_jobs=-1,
                eval_metric='mlogloss'
            )
            model.fit(X_train, y_train)
            best_params = model.get_params()
            best_score = model.score(X_test, y_test)
        
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted')
        acc = accuracy_score(y_test, y_pred)
        
        results = {
            'model': model,
            'best_params': best_params,
            'best_cv_score': float(best_score),
            'test_f1': float(f1),
            'test_accuracy': float(acc),
            'algorithm': 'XGBoost',
            'feature_importances': model.feature_importances_.tolist()
        }
        
        return model, results


class ModelEvaluator:
    """
    Comprehensive model evaluation and comparison.
    
    Purpose:
    - Calculate multiple evaluation metrics
    - Generate classification reports
    - Compare models across datasets
    - Visualize results
    
    Metrics Included:
    - Accuracy: Overall correctness
    - F1 Score: Balance between precision and recall
    - ROC-AUC: Discriminative ability (for binary)
    - Confusion Matrix: Classification breakdown
    - Cross-validation scores: Robustness assessment
    """
    
    @staticmethod
    def evaluate_model(
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive model evaluation.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            feature_names: Optional feature names for interpretation
            
        Returns:
            Dictionary with comprehensive metrics
        """
        y_pred = model.predict(X_test)
        
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'f1_weighted': float(f1_score(y_test, y_pred, average='weighted')),
            'f1_macro': float(f1_score(y_test, y_pred, average='macro')),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        # Feature importance if available
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            if feature_names:
                importance_dict = {name: float(imp) for name, imp in zip(feature_names, importances)}
                metrics['feature_importances'] = importance_dict
            else:
                metrics['feature_importances_array'] = importances.tolist()
        
        # Coefficients if available (logistic regression)
        if hasattr(model, 'coef_'):
            metrics['coefficients'] = model.coef_.tolist()
        
        return metrics
    
    @staticmethod
    def cross_validate_model(
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        cv: int = 5
    ) -> Dict[str, Any]:
        """
        Perform cross-validation evaluation.
        
        Args:
            model: Model to evaluate
            X_train: Training features
            y_train: Training labels
            cv: Number of folds
            
        Returns:
            Cross-validation scores and statistics
        """
        scorer = {'accuracy': 'accuracy', 'f1': 'f1_weighted'}
        
        from sklearn.model_selection import cross_validate
        cv_results = cross_validate(model, X_train, y_train, cv=cv, scoring=scorer)
        
        return {
            'accuracy_mean': float(cv_results['test_accuracy'].mean()),
            'accuracy_std': float(cv_results['test_accuracy'].std()),
            'f1_mean': float(cv_results['test_f1'].mean()),
            'f1_std': float(cv_results['test_f1'].std()),
            'fold_scores': {
                'accuracy': cv_results['test_accuracy'].tolist(),
                'f1': cv_results['test_f1'].tolist()
            }
        }
