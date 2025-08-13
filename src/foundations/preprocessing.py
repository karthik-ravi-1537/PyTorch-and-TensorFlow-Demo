"""
Preprocessing utilities for the ML frameworks tutorial.
Provides consistent data preprocessing across PyTorch and TensorFlow examples.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import re
import string

try:
    import torch
    import torch.nn.functional as F
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


class TextPreprocessor:
    """Text preprocessing utilities for NLP tasks."""
    
    def __init__(self, lowercase: bool = True, remove_punctuation: bool = True):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.vocab = {}
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab_size = 0
        
    def clean_text(self, text: str) -> str:
        """Clean individual text string."""
        if self.lowercase:
            text = text.lower()
        
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def build_vocabulary(self, texts: List[str], min_freq: int = 1, max_vocab_size: Optional[int] = None):
        """Build vocabulary from list of texts."""
        word_counts = {}
        
        for text in texts:
            cleaned_text = self.clean_text(text)
            words = cleaned_text.split()
            
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Filter by frequency
        filtered_words = {word: count for word, count in word_counts.items() if count >= min_freq}
        
        # Sort by frequency and limit vocabulary size
        sorted_words = sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)
        
        if max_vocab_size:
            sorted_words = sorted_words[:max_vocab_size]
        
        # Create mappings (reserve 0 for padding, 1 for unknown)
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx_to_word = {0: '<PAD>', 1: '<UNK>'}
        
        for i, (word, _) in enumerate(sorted_words, start=2):
            self.word_to_idx[word] = i
            self.idx_to_word[i] = word
        
        self.vocab_size = len(self.word_to_idx)
        self.vocab = {word: count for word, count in sorted_words}
        
        print(f"Built vocabulary with {self.vocab_size} words")
    
    def texts_to_sequences(self, texts: List[str], max_length: Optional[int] = None) -> List[List[int]]:
        """Convert texts to sequences of token indices."""
        sequences = []
        
        for text in texts:
            cleaned_text = self.clean_text(text)
            words = cleaned_text.split()
            
            sequence = [self.word_to_idx.get(word, 1) for word in words]  # 1 is <UNK>
            
            if max_length:
                if len(sequence) > max_length:
                    sequence = sequence[:max_length]
                else:
                    sequence.extend([0] * (max_length - len(sequence)))  # 0 is <PAD>
            
            sequences.append(sequence)
        
        return sequences
    
    def create_tfidf_features(self, texts: List[str], max_features: int = 5000) -> Tuple[np.ndarray, Any]:
        """Create TF-IDF features from texts."""
        cleaned_texts = [self.clean_text(text) for text in texts]
        
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        features = vectorizer.fit_transform(cleaned_texts).toarray()
        
        return features, vectorizer
    
    def create_bow_features(self, texts: List[str], max_features: int = 5000) -> Tuple[np.ndarray, Any]:
        """Create Bag of Words features from texts."""
        cleaned_texts = [self.clean_text(text) for text in texts]
        
        vectorizer = CountVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        features = vectorizer.fit_transform(cleaned_texts).toarray()
        
        return features, vectorizer


class TabularPreprocessor:
    """Preprocessing utilities for tabular data."""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        self.categorical_features = []
        self.numerical_features = []
    
    def identify_feature_types(self, df: pd.DataFrame, categorical_threshold: int = 10):
        """Automatically identify categorical and numerical features."""
        self.categorical_features = []
        self.numerical_features = []
        
        for column in df.columns:
            if df[column].dtype == 'object' or df[column].nunique() <= categorical_threshold:
                self.categorical_features.append(column)
            else:
                self.numerical_features.append(column)
        
        print(f"Identified {len(self.categorical_features)} categorical features: {self.categorical_features}")
        print(f"Identified {len(self.numerical_features)} numerical features: {self.numerical_features}")
    
    def scale_numerical_features(
        self, 
        X: np.ndarray, 
        feature_names: List[str],
        method: str = 'standard',
        fit: bool = True
    ) -> np.ndarray:
        """Scale numerical features."""
        if method == 'standard':
            scaler_class = StandardScaler
        elif method == 'minmax':
            scaler_class = MinMaxScaler
        else:
            raise ValueError("Method must be 'standard' or 'minmax'")
        
        X_scaled = X.copy()
        
        for i, feature_name in enumerate(feature_names):
            if feature_name in self.numerical_features:
                if fit:
                    scaler = scaler_class()
                    X_scaled[:, i:i+1] = scaler.fit_transform(X[:, i:i+1])
                    self.scalers[feature_name] = scaler
                else:
                    if feature_name in self.scalers:
                        X_scaled[:, i:i+1] = self.scalers[feature_name].transform(X[:, i:i+1])
        
        return X_scaled
    
    def encode_categorical_features(
        self, 
        X: np.ndarray, 
        feature_names: List[str],
        method: str = 'onehot',
        fit: bool = True
    ) -> Tuple[np.ndarray, List[str]]:
        """Encode categorical features."""
        if method not in ['onehot', 'label']:
            raise ValueError("Method must be 'onehot' or 'label'")
        
        X_encoded = []
        new_feature_names = []
        
        for i, feature_name in enumerate(feature_names):
            if feature_name in self.categorical_features:
                if method == 'onehot':
                    if fit:
                        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                        encoded = encoder.fit_transform(X[:, i:i+1])
                        self.encoders[feature_name] = encoder
                    else:
                        encoder = self.encoders[feature_name]
                        encoded = encoder.transform(X[:, i:i+1])
                    
                    # Create feature names for one-hot encoded columns
                    categories = encoder.categories_[0]
                    for category in categories:
                        new_feature_names.append(f"{feature_name}_{category}")
                    
                    X_encoded.append(encoded)
                
                elif method == 'label':
                    if fit:
                        encoder = LabelEncoder()
                        encoded = encoder.fit_transform(X[:, i]).reshape(-1, 1)
                        self.encoders[feature_name] = encoder
                    else:
                        encoder = self.encoders[feature_name]
                        encoded = encoder.transform(X[:, i]).reshape(-1, 1)
                    
                    new_feature_names.append(feature_name)
                    X_encoded.append(encoded)
            
            else:
                # Numerical feature - keep as is
                X_encoded.append(X[:, i:i+1])
                new_feature_names.append(feature_name)
        
        return np.concatenate(X_encoded, axis=1), new_feature_names
    
    def preprocess_tabular_data(
        self, 
        X: Union[np.ndarray, pd.DataFrame], 
        feature_names: Optional[List[str]] = None,
        scaling_method: str = 'standard',
        encoding_method: str = 'onehot',
        fit: bool = True
    ) -> Tuple[np.ndarray, List[str]]:
        """Complete preprocessing pipeline for tabular data."""
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            if feature_names is None:
                feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            df = pd.DataFrame(X, columns=feature_names)
        else:
            df = X.copy()
            feature_names = list(df.columns)
        
        # Identify feature types
        if fit:
            self.identify_feature_types(df)
            self.feature_names = feature_names
        
        # Convert back to numpy for processing
        X_array = df.values
        
        # Encode categorical features
        X_encoded, new_feature_names = self.encode_categorical_features(
            X_array, feature_names, encoding_method, fit
        )
        
        # Scale numerical features
        X_processed = self.scale_numerical_features(
            X_encoded, new_feature_names, scaling_method, fit
        )
        
        return X_processed, new_feature_names


class TimeSeriesPreprocessor:
    """Preprocessing utilities for time series data."""
    
    def __init__(self):
        self.scalers = {}
        self.feature_stats = {}
    
    def normalize_series(
        self, 
        series: np.ndarray, 
        method: str = 'standard',
        fit: bool = True,
        feature_axis: int = -1
    ) -> np.ndarray:
        """Normalize time series data."""
        if method == 'standard':
            if fit:
                mean = np.mean(series, axis=feature_axis, keepdims=True)
                std = np.std(series, axis=feature_axis, keepdims=True)
                self.feature_stats['mean'] = mean
                self.feature_stats['std'] = std
            else:
                mean = self.feature_stats['mean']
                std = self.feature_stats['std']
            
            return (series - mean) / (std + 1e-8)
        
        elif method == 'minmax':
            if fit:
                min_val = np.min(series, axis=feature_axis, keepdims=True)
                max_val = np.max(series, axis=feature_axis, keepdims=True)
                self.feature_stats['min'] = min_val
                self.feature_stats['max'] = max_val
            else:
                min_val = self.feature_stats['min']
                max_val = self.feature_stats['max']
            
            return (series - min_val) / (max_val - min_val + 1e-8)
        
        else:
            raise ValueError("Method must be 'standard' or 'minmax'")
    
    def create_sequences(
        self, 
        data: np.ndarray, 
        sequence_length: int,
        prediction_horizon: int = 1,
        step_size: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create input sequences and targets for time series prediction."""
        X, y = [], []
        
        for i in range(0, len(data) - sequence_length - prediction_horizon + 1, step_size):
            # Input sequence
            X.append(data[i:i + sequence_length])
            
            # Target (next value(s))
            if prediction_horizon == 1:
                y.append(data[i + sequence_length])
            else:
                y.append(data[i + sequence_length:i + sequence_length + prediction_horizon])
        
        return np.array(X), np.array(y)
    
    def add_time_features(self, timestamps: np.ndarray) -> np.ndarray:
        """Add time-based features (hour, day, month, etc.)."""
        # Convert to pandas datetime if needed
        if not isinstance(timestamps, pd.DatetimeIndex):
            timestamps = pd.to_datetime(timestamps)
        
        time_features = np.column_stack([
            timestamps.hour,
            timestamps.day,
            timestamps.month,
            timestamps.dayofweek,
            timestamps.dayofyear
        ])
        
        return time_features
    
    def handle_missing_values(
        self, 
        series: np.ndarray, 
        method: str = 'interpolate'
    ) -> np.ndarray:
        """Handle missing values in time series."""
        if method == 'interpolate':
            # Linear interpolation
            df = pd.DataFrame(series)
            return df.interpolate().values
        
        elif method == 'forward_fill':
            df = pd.DataFrame(series)
            return df.fillna(method='ffill').values
        
        elif method == 'backward_fill':
            df = pd.DataFrame(series)
            return df.fillna(method='bfill').values
        
        elif method == 'mean':
            mean_val = np.nanmean(series, axis=0)
            series_filled = series.copy()
            series_filled[np.isnan(series_filled)] = mean_val
            return series_filled
        
        else:
            raise ValueError("Method must be 'interpolate', 'forward_fill', 'backward_fill', or 'mean'")


def create_pytorch_embeddings(vocab_size: int, embedding_dim: int, padding_idx: int = 0):
    """Create PyTorch embedding layer."""
    if not PYTORCH_AVAILABLE:
        raise ImportError("PyTorch not available")
    
    return torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)


def create_tensorflow_embeddings(vocab_size: int, embedding_dim: int, mask_zero: bool = True):
    """Create TensorFlow embedding layer."""
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow not available")
    
    return tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=mask_zero)


def pad_sequences_pytorch(sequences: List[List[int]], max_length: Optional[int] = None, padding_value: int = 0):
    """Pad sequences for PyTorch (similar to tf.keras.preprocessing.sequence.pad_sequences)."""
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)
    
    padded = []
    for seq in sequences:
        if len(seq) > max_length:
            padded.append(seq[:max_length])
        else:
            padded.append(seq + [padding_value] * (max_length - len(seq)))
    
    return np.array(padded)


def pad_sequences_tensorflow(sequences: List[List[int]], max_length: Optional[int] = None, padding: str = 'post'):
    """Pad sequences for TensorFlow."""
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow not available")
    
    return tf.keras.preprocessing.sequence.pad_sequences(
        sequences, maxlen=max_length, padding=padding
    )


# Convenience functions
def preprocess_text_data(texts: List[str], max_vocab_size: int = 10000, max_length: int = 100):
    """Quick text preprocessing for both frameworks."""
    preprocessor = TextPreprocessor()
    preprocessor.build_vocabulary(texts, max_vocab_size=max_vocab_size)
    sequences = preprocessor.texts_to_sequences(texts, max_length=max_length)
    
    return {
        'sequences': np.array(sequences),
        'vocab_size': preprocessor.vocab_size,
        'word_to_idx': preprocessor.word_to_idx,
        'idx_to_word': preprocessor.idx_to_word,
        'preprocessor': preprocessor
    }


def preprocess_tabular_data(X, feature_names=None, scaling='standard', encoding='onehot'):
    """Quick tabular data preprocessing."""
    preprocessor = TabularPreprocessor()
    X_processed, new_feature_names = preprocessor.preprocess_tabular_data(
        X, feature_names, scaling, encoding
    )
    
    return {
        'X_processed': X_processed,
        'feature_names': new_feature_names,
        'preprocessor': preprocessor
    }


def preprocess_timeseries_data(series, sequence_length=50, normalize=True):
    """Quick time series preprocessing."""
    preprocessor = TimeSeriesPreprocessor()
    
    if normalize:
        series_normalized = preprocessor.normalize_series(series)
    else:
        series_normalized = series
    
    X, y = preprocessor.create_sequences(series_normalized, sequence_length)
    
    return {
        'X': X,
        'y': y,
        'preprocessor': preprocessor
    }