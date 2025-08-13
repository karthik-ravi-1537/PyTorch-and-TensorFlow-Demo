"""
Data loading utilities for the ML frameworks tutorial.
Provides consistent data loading and management across PyTorch and TensorFlow examples.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import os
import urllib.request
import zipfile
import json
from pathlib import Path

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


class DatasetManager:
    """Main class for managing datasets across different domains."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.data_dir / "sample_datasets").mkdir(exist_ok=True)
        (self.data_dir / "nlp").mkdir(exist_ok=True)
        (self.data_dir / "tabular").mkdir(exist_ok=True)
        (self.data_dir / "timeseries").mkdir(exist_ok=True)
    
    def get_sample_text_data(self, num_samples: int = 1000) -> Dict[str, Any]:
        """
        Generate sample text data for NLP examples.
        
        Args:
            num_samples: Number of text samples to generate
            
        Returns:
            Dictionary containing text data and labels
        """
        # Sample text categories and templates
        categories = ['positive', 'negative', 'neutral']
        
        positive_templates = [
            "This is an excellent product that I highly recommend.",
            "Amazing quality and great customer service.",
            "I love this item, it exceeded my expectations.",
            "Outstanding performance and value for money.",
            "Fantastic experience, will definitely buy again."
        ]
        
        negative_templates = [
            "This product is terrible and not worth the money.",
            "Poor quality and disappointing performance.",
            "I regret buying this item, it's completely useless.",
            "Awful customer service and defective product.",
            "Waste of money, would not recommend to anyone."
        ]
        
        neutral_templates = [
            "This product is okay, nothing special about it.",
            "Average quality for the price point.",
            "It works as expected, no complaints or praise.",
            "Standard product with typical features.",
            "Decent item but there are better alternatives."
        ]
        
        templates = {
            'positive': positive_templates,
            'negative': negative_templates,
            'neutral': neutral_templates
        }
        
        texts = []
        labels = []
        
        np.random.seed(42)  # For reproducibility
        
        for _ in range(num_samples):
            category = np.random.choice(categories)
            template = np.random.choice(templates[category])
            
            # Add some variation
            variations = [
                template,
                template.replace("product", "item"),
                template.replace("This", "The"),
                f"In my opinion, {template.lower()}",
                f"Overall, {template.lower()}"
            ]
            
            text = np.random.choice(variations)
            texts.append(text)
            labels.append(categories.index(category))
        
        return {
            'texts': texts,
            'labels': labels,
            'label_names': categories,
            'num_classes': len(categories)
        }
    
    def get_sample_tabular_data(
        self, 
        num_samples: int = 1000, 
        num_features: int = 10,
        task_type: str = 'classification'
    ) -> Dict[str, Any]:
        """
        Generate sample tabular data for structured data examples.
        
        Args:
            num_samples: Number of samples to generate
            num_features: Number of features
            task_type: 'classification' or 'regression'
            
        Returns:
            Dictionary containing tabular data and targets
        """
        np.random.seed(42)
        
        # Generate feature data
        X = np.random.randn(num_samples, num_features)
        
        # Add some correlation structure
        for i in range(1, num_features):
            X[:, i] += 0.3 * X[:, i-1] + 0.1 * np.random.randn(num_samples)
        
        # Generate targets based on task type
        if task_type == 'classification':
            # Create non-linear decision boundary
            decision_score = (
                2 * X[:, 0] + 
                1.5 * X[:, 1] - 
                0.5 * X[:, 2] + 
                0.3 * X[:, 0] * X[:, 1] +
                np.random.randn(num_samples) * 0.1
            )
            y = (decision_score > 0).astype(int)
            
            return {
                'X': X,
                'y': y,
                'feature_names': [f'feature_{i}' for i in range(num_features)],
                'target_names': ['class_0', 'class_1'],
                'task_type': 'classification',
                'num_classes': 2
            }
        
        else:  # regression
            # Create non-linear relationship
            y = (
                2 * X[:, 0] + 
                1.5 * X[:, 1] - 
                0.5 * X[:, 2] + 
                0.3 * X[:, 0] * X[:, 1] +
                0.1 * X[:, 0] ** 2 +
                np.random.randn(num_samples) * 0.2
            )
            
            return {
                'X': X,
                'y': y,
                'feature_names': [f'feature_{i}' for i in range(num_features)],
                'task_type': 'regression'
            }
    
    def get_sample_timeseries_data(
        self, 
        num_series: int = 100,
        series_length: int = 100,
        num_features: int = 1
    ) -> Dict[str, Any]:
        """
        Generate sample time series data.
        
        Args:
            num_series: Number of time series
            series_length: Length of each series
            num_features: Number of features per time step
            
        Returns:
            Dictionary containing time series data
        """
        np.random.seed(42)
        
        # Generate time series with trend, seasonality, and noise
        time_series = []
        targets = []
        
        for i in range(num_series):
            # Base trend
            trend = np.linspace(0, 2, series_length) * np.random.uniform(0.5, 2.0)
            
            # Seasonal component
            seasonal = np.sin(2 * np.pi * np.arange(series_length) / 20) * np.random.uniform(0.5, 1.5)
            
            # Noise
            noise = np.random.randn(series_length) * 0.1
            
            # Combine components
            if num_features == 1:
                series = trend + seasonal + noise
                series = series.reshape(-1, 1)
            else:
                # Multiple features with different patterns
                series = np.zeros((series_length, num_features))
                series[:, 0] = trend + seasonal + noise
                
                for j in range(1, num_features):
                    phase_shift = np.random.uniform(0, 2*np.pi)
                    feature_seasonal = np.sin(2 * np.pi * np.arange(series_length) / 15 + phase_shift)
                    feature_trend = np.linspace(0, 1, series_length) * np.random.uniform(-1, 1)
                    feature_noise = np.random.randn(series_length) * 0.05
                    
                    series[:, j] = feature_trend + feature_seasonal + feature_noise
            
            time_series.append(series)
            
            # Target: predict next value or classify trend direction
            next_value = trend[-1] + seasonal[-1] + np.random.randn() * 0.1
            targets.append(next_value)
        
        return {
            'time_series': np.array(time_series),
            'targets': np.array(targets),
            'series_length': series_length,
            'num_features': num_features,
            'num_series': num_series
        }
    
    def create_train_val_test_split(
        self, 
        data: Dict[str, Any], 
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_state: int = 42
    ) -> Dict[str, Dict[str, Any]]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            data: Data dictionary from get_sample_* methods
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with train, val, test splits
        """
        np.random.seed(random_state)
        
        # Determine the main data arrays
        if 'texts' in data:
            # Text data
            X = data['texts']
            y = data['labels']
        elif 'time_series' in data:
            # Time series data
            X = data['time_series']
            y = data['targets']
        else:
            # Tabular data
            X = data['X']
            y = data['y']
        
        # Create indices for splitting
        n_samples = len(X)
        indices = np.random.permutation(n_samples)
        
        train_end = int(train_ratio * n_samples)
        val_end = train_end + int(val_ratio * n_samples)
        
        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]
        
        # Create splits
        splits = {}
        
        for split_name, idx in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
            if isinstance(X, list):
                split_X = [X[i] for i in idx]
            else:
                split_X = X[idx]
            
            if isinstance(y, list):
                split_y = [y[i] for i in idx]
            else:
                split_y = y[idx]
            
            splits[split_name] = {
                'X': split_X,
                'y': split_y,
                'size': len(idx)
            }
            
            # Copy metadata
            for key, value in data.items():
                if key not in ['texts', 'labels', 'X', 'y', 'time_series', 'targets']:
                    splits[split_name][key] = value
        
        return splits
    
    def save_dataset(self, data: Dict[str, Any], filename: str, domain: str = "sample_datasets"):
        """Save dataset to disk."""
        filepath = self.data_dir / domain / f"{filename}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_data = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                serializable_data[key] = value.tolist()
            else:
                serializable_data[key] = value
        
        with open(filepath, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        
        print(f"Dataset saved to {filepath}")
    
    def load_dataset(self, filename: str, domain: str = "sample_datasets") -> Dict[str, Any]:
        """Load dataset from disk."""
        filepath = self.data_dir / domain / f"{filename}.json"
        
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset not found: {filepath}")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Convert lists back to numpy arrays where appropriate
        array_keys = ['X', 'y', 'labels', 'time_series', 'targets']
        for key in array_keys:
            if key in data:
                data[key] = np.array(data[key])
        
        return data


class PyTorchDataset(Dataset):
    """PyTorch Dataset wrapper for tutorial data."""
    
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        sample = self.X[idx]
        target = self.y[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        # Convert to tensors
        if isinstance(sample, np.ndarray):
            sample = torch.FloatTensor(sample)
        elif isinstance(sample, str):
            # For text data, return as string (will be tokenized later)
            pass
        
        if isinstance(target, np.ndarray):
            target = torch.FloatTensor(target)
        else:
            target = torch.LongTensor([target]) if isinstance(target, int) else torch.FloatTensor([target])
        
        return sample, target


def create_pytorch_dataloader(
    data_split: Dict[str, Any], 
    batch_size: int = 32, 
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """Create PyTorch DataLoader from data split."""
    if not PYTORCH_AVAILABLE:
        raise ImportError("PyTorch not available")
    
    dataset = PyTorchDataset(data_split['X'], data_split['y'])
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def create_tensorflow_dataset(
    data_split: Dict[str, Any], 
    batch_size: int = 32, 
    shuffle: bool = True,
    buffer_size: int = 1000
):
    """Create TensorFlow Dataset from data split."""
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow not available")
    
    X = data_split['X']
    y = data_split['y']
    
    # Handle different data types
    if isinstance(X[0], str):
        # Text data
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
    else:
        # Numerical data
        X_tensor = tf.constant(X, dtype=tf.float32)
        y_tensor = tf.constant(y, dtype=tf.int64 if data_split.get('task_type') == 'classification' else tf.float32)
        dataset = tf.data.Dataset.from_tensor_slices((X_tensor, y_tensor))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size)
    
    dataset = dataset.batch(batch_size)
    return dataset


# Convenience functions
def get_tutorial_text_data(num_samples: int = 1000):
    """Quick access to sample text data."""
    manager = DatasetManager()
    return manager.get_sample_text_data(num_samples)


def get_tutorial_tabular_data(
    num_samples: int = 1000, 
    task_type: str = 'classification',
    return_as_dataframe: bool = False,
    include_categorical: bool = False
):
    """Quick access to sample tabular data."""
    manager = DatasetManager()
    data = manager.get_sample_tabular_data(num_samples, task_type=task_type)
    
    if return_as_dataframe:
        # Convert to DataFrame
        df_data = {}
        
        # Add features
        for i, feature_name in enumerate(data['feature_names']):
            df_data[feature_name] = data['X'][:, i]
        
        # Add categorical features if requested
        if include_categorical:
            # Add some categorical features
            np.random.seed(42)
            df_data['category_A'] = np.random.choice(['Type1', 'Type2', 'Type3'], size=num_samples)
            df_data['category_B'] = np.random.choice(['Small', 'Medium', 'Large'], size=num_samples)
            df_data['binary_feature'] = np.random.choice([0, 1], size=num_samples)
        
        # Add target
        df_data['target'] = data['y']
        
        # Create DataFrame
        df = pd.DataFrame(df_data)
        
        # Return data with DataFrame
        result = data.copy()
        result['dataframe'] = df
        return result
    
    return data


def get_tutorial_timeseries_data(num_series: int = 100, series_length: int = 100):
    """Quick access to sample time series data."""
    manager = DatasetManager()
    return manager.get_sample_timeseries_data(num_series, series_length)


def create_framework_datasets(data_split: Dict[str, Any], batch_size: int = 32):
    """Create both PyTorch and TensorFlow datasets from the same data split."""
    datasets = {}
    
    if PYTORCH_AVAILABLE:
        try:
            datasets['pytorch'] = create_pytorch_dataloader(data_split, batch_size)
        except Exception as e:
            print(f"Failed to create PyTorch dataset: {e}")
    
    if TENSORFLOW_AVAILABLE:
        try:
            datasets['tensorflow'] = create_tensorflow_dataset(data_split, batch_size)
        except Exception as e:
            print(f"Failed to create TensorFlow dataset: {e}")
    
    return datasets