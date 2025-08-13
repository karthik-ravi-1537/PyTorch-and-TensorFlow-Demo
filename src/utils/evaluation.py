"""
Evaluation utilities for comparing PyTorch and TensorFlow model performance.
Provides framework-agnostic metrics, benchmarking tools, and evaluation helpers.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import time
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report
)

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


class FrameworkEvaluator:
    """Main class for evaluating and comparing model performance across frameworks."""
    
    def __init__(self):
        self.pytorch_available = PYTORCH_AVAILABLE
        self.tensorflow_available = TENSORFLOW_AVAILABLE
        self.evaluation_history = {}
    
    def evaluate_classification(
        self,
        y_true: np.ndarray,
        pytorch_predictions: Optional[np.ndarray] = None,
        tensorflow_predictions: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None,
        average: str = 'weighted'
    ) -> Dict[str, Any]:
        """
        Evaluate classification performance for both frameworks.
        
        Args:
            y_true: True labels
            pytorch_predictions: PyTorch model predictions
            tensorflow_predictions: TensorFlow model predictions
            class_names: Optional class names for reporting
            average: Averaging strategy for multi-class metrics
            
        Returns:
            Dictionary containing evaluation results for both frameworks
        """
        results = {
            'task_type': 'classification',
            'pytorch': {'available': self.pytorch_available},
            'tensorflow': {'available': self.tensorflow_available},
            'comparison': {}
        }
        
        # Evaluate PyTorch predictions
        if pytorch_predictions is not None:
            results['pytorch'].update(
                self._evaluate_single_classification(
                    y_true, pytorch_predictions, class_names, average
                )
            )
        
        # Evaluate TensorFlow predictions
        if tensorflow_predictions is not None:
            results['tensorflow'].update(
                self._evaluate_single_classification(
                    y_true, tensorflow_predictions, class_names, average
                )
            )
        
        # Compare results
        if (pytorch_predictions is not None and 
            tensorflow_predictions is not None):
            results['comparison'] = self._compare_predictions(
                pytorch_predictions, tensorflow_predictions
            )
        
        return results
    
    def evaluate_regression(
        self,
        y_true: np.ndarray,
        pytorch_predictions: Optional[np.ndarray] = None,
        tensorflow_predictions: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Evaluate regression performance for both frameworks.
        
        Args:
            y_true: True values
            pytorch_predictions: PyTorch model predictions
            tensorflow_predictions: TensorFlow model predictions
            
        Returns:
            Dictionary containing evaluation results for both frameworks
        """
        results = {
            'task_type': 'regression',
            'pytorch': {'available': self.pytorch_available},
            'tensorflow': {'available': self.tensorflow_available},
            'comparison': {}
        }
        
        # Evaluate PyTorch predictions
        if pytorch_predictions is not None:
            results['pytorch'].update(
                self._evaluate_single_regression(y_true, pytorch_predictions)
            )
        
        # Evaluate TensorFlow predictions
        if tensorflow_predictions is not None:
            results['tensorflow'].update(
                self._evaluate_single_regression(y_true, tensorflow_predictions)
            )
        
        # Compare results
        if (pytorch_predictions is not None and 
            tensorflow_predictions is not None):
            results['comparison'] = self._compare_predictions(
                pytorch_predictions, tensorflow_predictions
            )
        
        return results
    
    def benchmark_inference_speed(
        self,
        pytorch_model=None,
        tensorflow_model=None,
        sample_input=None,
        num_runs: int = 100,
        warmup_runs: int = 10
    ) -> Dict[str, Any]:
        """
        Benchmark inference speed for both frameworks.
        
        Args:
            pytorch_model: PyTorch model
            tensorflow_model: TensorFlow model
            sample_input: Sample input for inference
            num_runs: Number of inference runs
            warmup_runs: Number of warmup runs
            
        Returns:
            Benchmark results for both frameworks
        """
        results = {
            'pytorch': {'available': self.pytorch_available},
            'tensorflow': {'available': self.tensorflow_available},
            'comparison': {}
        }
        
        # Benchmark PyTorch
        if pytorch_model is not None and self.pytorch_available:
            results['pytorch'].update(
                self._benchmark_pytorch_inference(
                    pytorch_model, sample_input, num_runs, warmup_runs
                )
            )
        
        # Benchmark TensorFlow
        if tensorflow_model is not None and self.tensorflow_available:
            results['tensorflow'].update(
                self._benchmark_tensorflow_inference(
                    tensorflow_model, sample_input, num_runs, warmup_runs
                )
            )
        
        # Compare performance
        if (results['pytorch'].get('mean_time') and 
            results['tensorflow'].get('mean_time')):
            pt_time = results['pytorch']['mean_time']
            tf_time = results['tensorflow']['mean_time']
            
            results['comparison'] = {
                'pytorch_faster': pt_time < tf_time,
                'speedup_ratio': max(pt_time, tf_time) / min(pt_time, tf_time),
                'faster_framework': 'pytorch' if pt_time < tf_time else 'tensorflow',
                'time_difference': abs(pt_time - tf_time)
            }
        
        return results
    
    def evaluate_memory_usage(
        self,
        pytorch_model=None,
        tensorflow_model=None,
        sample_input=None
    ) -> Dict[str, Any]:
        """
        Evaluate memory usage for both frameworks.
        
        Args:
            pytorch_model: PyTorch model
            tensorflow_model: TensorFlow model
            sample_input: Sample input
            
        Returns:
            Memory usage comparison
        """
        results = {
            'pytorch': {'available': self.pytorch_available},
            'tensorflow': {'available': self.tensorflow_available}
        }
        
        # PyTorch memory usage
        if pytorch_model is not None and self.pytorch_available:
            results['pytorch'].update(
                self._measure_pytorch_memory(pytorch_model, sample_input)
            )
        
        # TensorFlow memory usage
        if tensorflow_model is not None and self.tensorflow_available:
            results['tensorflow'].update(
                self._measure_tensorflow_memory(tensorflow_model, sample_input)
            )
        
        return results
    
    def _evaluate_single_classification(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        class_names: Optional[List[str]], 
        average: str
    ) -> Dict[str, Any]:
        """Evaluate classification metrics for a single framework."""
        try:
            # Convert predictions to class labels if they're probabilities
            if y_pred.ndim > 1 and y_pred.shape[1] > 1:
                y_pred_labels = np.argmax(y_pred, axis=1)
            else:
                y_pred_labels = y_pred.flatten()
            
            # Calculate metrics
            metrics = {
                'accuracy': float(accuracy_score(y_true, y_pred_labels)),
                'precision': float(precision_score(y_true, y_pred_labels, average=average, zero_division=0)),
                'recall': float(recall_score(y_true, y_pred_labels, average=average, zero_division=0)),
                'f1_score': float(f1_score(y_true, y_pred_labels, average=average, zero_division=0)),
                'confusion_matrix': confusion_matrix(y_true, y_pred_labels).tolist(),
                'classification_report': classification_report(
                    y_true, y_pred_labels, 
                    target_names=class_names, 
                    output_dict=True,
                    zero_division=0
                )
            }
            
            return {'success': True, 'metrics': metrics}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _evaluate_single_regression(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Evaluate regression metrics for a single framework."""
        try:
            # Flatten predictions if needed
            y_pred_flat = y_pred.flatten() if y_pred.ndim > 1 else y_pred
            y_true_flat = y_true.flatten() if y_true.ndim > 1 else y_true
            
            metrics = {
                'mse': float(mean_squared_error(y_true_flat, y_pred_flat)),
                'rmse': float(np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))),
                'mae': float(mean_absolute_error(y_true_flat, y_pred_flat)),
                'r2_score': float(r2_score(y_true_flat, y_pred_flat)),
                'mean_residual': float(np.mean(y_true_flat - y_pred_flat)),
                'std_residual': float(np.std(y_true_flat - y_pred_flat))
            }
            
            return {'success': True, 'metrics': metrics}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _compare_predictions(
        self, 
        pytorch_pred: np.ndarray, 
        tensorflow_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Compare predictions between frameworks."""
        try:
            # Flatten for comparison
            pt_flat = pytorch_pred.flatten()
            tf_flat = tensorflow_pred.flatten()
            
            if len(pt_flat) != len(tf_flat):
                return {'error': 'Prediction shapes do not match'}
            
            # Calculate differences
            diff = np.abs(pt_flat - tf_flat)
            
            comparison = {
                'max_difference': float(np.max(diff)),
                'mean_difference': float(np.mean(diff)),
                'std_difference': float(np.std(diff)),
                'predictions_close': bool(np.allclose(pt_flat, tf_flat, rtol=1e-5)),
                'correlation': float(np.corrcoef(pt_flat, tf_flat)[0, 1]) if len(pt_flat) > 1 else 1.0
            }
            
            return comparison
            
        except Exception as e:
            return {'error': str(e)}
    
    def _benchmark_pytorch_inference(
        self, 
        model, 
        sample_input, 
        num_runs: int, 
        warmup_runs: int
    ) -> Dict[str, Any]:
        """Benchmark PyTorch model inference."""
        if not self.pytorch_available:
            return {'success': False, 'error': 'PyTorch not available'}
        
        try:
            model.eval()
            times = []
            
            # Warmup runs
            with torch.no_grad():
                for _ in range(warmup_runs):
                    _ = model(sample_input)
            
            # Actual timing runs
            with torch.no_grad():
                for _ in range(num_runs):
                    start_time = time.time()
                    _ = model(sample_input)
                    end_time = time.time()
                    times.append(end_time - start_time)
            
            return {
                'success': True,
                'times': times,
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'min_time': np.min(times),
                'max_time': np.max(times),
                'throughput': num_runs / np.sum(times)  # samples per second
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _benchmark_tensorflow_inference(
        self, 
        model, 
        sample_input, 
        num_runs: int, 
        warmup_runs: int
    ) -> Dict[str, Any]:
        """Benchmark TensorFlow model inference."""
        if not self.tensorflow_available:
            return {'success': False, 'error': 'TensorFlow not available'}
        
        try:
            times = []
            
            # Warmup runs
            for _ in range(warmup_runs):
                _ = model(sample_input, training=False)
            
            # Actual timing runs
            for _ in range(num_runs):
                start_time = time.time()
                _ = model(sample_input, training=False)
                end_time = time.time()
                times.append(end_time - start_time)
            
            return {
                'success': True,
                'times': times,
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'min_time': np.min(times),
                'max_time': np.max(times),
                'throughput': num_runs / np.sum(times)  # samples per second
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _measure_pytorch_memory(self, model, sample_input) -> Dict[str, Any]:
        """Measure PyTorch model memory usage."""
        if not self.pytorch_available:
            return {'success': False, 'error': 'PyTorch not available'}
        
        try:
            # Model parameters memory
            param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
            
            # Input memory
            input_memory = sample_input.numel() * sample_input.element_size()
            
            # Forward pass memory (approximate)
            model.eval()
            with torch.no_grad():
                output = model(sample_input)
                output_memory = output.numel() * output.element_size()
            
            return {
                'success': True,
                'parameter_memory_bytes': param_memory,
                'input_memory_bytes': input_memory,
                'output_memory_bytes': output_memory,
                'total_memory_mb': (param_memory + input_memory + output_memory) / (1024 * 1024)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _measure_tensorflow_memory(self, model, sample_input) -> Dict[str, Any]:
        """Measure TensorFlow model memory usage."""
        if not self.tensorflow_available:
            return {'success': False, 'error': 'TensorFlow not available'}
        
        try:
            # Model parameters memory
            param_memory = sum(np.prod(var.shape) * 4 for var in model.trainable_variables)  # Assume float32
            
            # Input memory
            input_memory = np.prod(sample_input.shape) * 4  # Assume float32
            
            # Forward pass memory (approximate)
            output = model(sample_input, training=False)
            output_memory = np.prod(output.shape) * 4  # Assume float32
            
            return {
                'success': True,
                'parameter_memory_bytes': param_memory,
                'input_memory_bytes': input_memory,
                'output_memory_bytes': output_memory,
                'total_memory_mb': (param_memory + input_memory + output_memory) / (1024 * 1024)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def create_evaluation_report(
        self, 
        evaluation_results: Dict[str, Any],
        title: str = "Framework Evaluation Report"
    ) -> str:
        """Create a formatted evaluation report."""
        report = [f"\n{title}", "=" * len(title)]
        
        task_type = evaluation_results.get('task_type', 'unknown')
        report.append(f"Task Type: {task_type.capitalize()}")
        
        # PyTorch results
        pt_results = evaluation_results.get('pytorch', {})
        report.append(f"\n🔥 PyTorch Results:")
        if pt_results.get('success'):
            metrics = pt_results.get('metrics', {})
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    report.append(f"   {metric}: {value:.4f}")
        else:
            report.append(f"   ❌ Error: {pt_results.get('error', 'Unknown error')}")
        
        # TensorFlow results
        tf_results = evaluation_results.get('tensorflow', {})
        report.append(f"\n🟠 TensorFlow Results:")
        if tf_results.get('success'):
            metrics = tf_results.get('metrics', {})
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    report.append(f"   {metric}: {value:.4f}")
        else:
            report.append(f"   ❌ Error: {tf_results.get('error', 'Unknown error')}")
        
        # Comparison
        comparison = evaluation_results.get('comparison', {})
        if comparison and not comparison.get('error'):
            report.append(f"\n📊 Comparison:")
            for metric, value in comparison.items():
                if isinstance(value, (int, float)):
                    report.append(f"   {metric}: {value:.4f}")
                elif isinstance(value, bool):
                    report.append(f"   {metric}: {value}")
        
        return "\n".join(report)


# Convenience functions
def quick_classification_eval(y_true, pytorch_pred=None, tensorflow_pred=None):
    """Quick classification evaluation."""
    evaluator = FrameworkEvaluator()
    return evaluator.evaluate_classification(y_true, pytorch_pred, tensorflow_pred)


def quick_regression_eval(y_true, pytorch_pred=None, tensorflow_pred=None):
    """Quick regression evaluation."""
    evaluator = FrameworkEvaluator()
    return evaluator.evaluate_regression(y_true, pytorch_pred, tensorflow_pred)


def quick_inference_benchmark(pytorch_model=None, tensorflow_model=None, sample_input=None):
    """Quick inference speed benchmark."""
    evaluator = FrameworkEvaluator()
    return evaluator.benchmark_inference_speed(pytorch_model, tensorflow_model, sample_input)