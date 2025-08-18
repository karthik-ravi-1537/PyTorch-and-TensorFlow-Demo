"""
Utilities for comparing PyTorch and TensorFlow implementations side-by-side.
Provides tools for code execution, performance benchmarking, and result comparison.
"""

import time
import traceback
from collections.abc import Callable
from typing import Any

import numpy as np

try:
    import torch

    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

try:
    import tensorflow  # noqa: F401

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


class FrameworkComparison:
    """Main class for comparing PyTorch and TensorFlow implementations."""

    def __init__(self):
        self.pytorch_available = PYTORCH_AVAILABLE
        self.tensorflow_available = TENSORFLOW_AVAILABLE
        self.results = {}

    def compare_implementations(
        self,
        pytorch_func: Callable | None = None,
        tensorflow_func: Callable | None = None,
        inputs: dict[str, Any] = None,
        name: str = "comparison",
    ) -> dict[str, Any]:
        """
        Compare PyTorch and TensorFlow implementations with the same inputs.

        Args:
            pytorch_func: Function implementing PyTorch version
            tensorflow_func: Function implementing TensorFlow version
            inputs: Dictionary of input parameters
            name: Name for this comparison

        Returns:
            Dictionary containing results, timing, and comparison metrics
        """
        results = {
            "name": name,
            "pytorch": {"available": self.pytorch_available},
            "tensorflow": {"available": self.tensorflow_available},
            "comparison": {},
        }

        # Run PyTorch implementation
        if pytorch_func and self.pytorch_available:
            results["pytorch"].update(self._run_implementation(pytorch_func, inputs, "pytorch"))

        # Run TensorFlow implementation
        if tensorflow_func and self.tensorflow_available:
            results["tensorflow"].update(self._run_implementation(tensorflow_func, inputs, "tensorflow"))

        # Compare results if both succeeded
        if results["pytorch"].get("success") and results["tensorflow"].get("success"):
            results["comparison"] = self._compare_results(results["pytorch"]["output"], results["tensorflow"]["output"])

        self.results[name] = results
        return results

    def _run_implementation(self, func: Callable, inputs: dict[str, Any], framework: str) -> dict[str, Any]:
        """Run a single implementation and capture results and timing."""
        result = {"success": False, "execution_time": 0.0, "memory_usage": 0.0, "output": None, "error": None}

        try:
            # Measure execution time
            start_time = time.time()

            # Run the function
            if inputs:
                output = func(**inputs)
            else:
                output = func()

            end_time = time.time()

            result.update({"success": True, "execution_time": end_time - start_time, "output": output})

            # Try to measure memory usage for tensors
            if framework == "pytorch" and hasattr(output, "element_size"):
                result["memory_usage"] = output.numel() * output.element_size()
            elif framework == "tensorflow" and hasattr(output, "numpy"):
                result["memory_usage"] = output.numpy().nbytes

        except Exception as e:
            result.update({"success": False, "error": str(e), "traceback": traceback.format_exc()})

        return result

    def _compare_results(self, pytorch_output: Any, tensorflow_output: Any) -> dict[str, Any]:
        """Compare outputs from PyTorch and TensorFlow implementations."""
        comparison = {
            "outputs_equal": False,
            "max_difference": None,
            "mean_difference": None,
            "shape_match": False,
            "dtype_match": False,
        }

        try:
            # Convert to numpy for comparison
            pt_array = self._to_numpy(pytorch_output)
            tf_array = self._to_numpy(tensorflow_output)

            if pt_array is not None and tf_array is not None:
                # Check shapes
                comparison["shape_match"] = pt_array.shape == tf_array.shape

                # Check dtypes (approximately)
                comparison["dtype_match"] = str(pt_array.dtype) == str(tf_array.dtype)

                if comparison["shape_match"]:
                    # Calculate differences
                    diff = np.abs(pt_array - tf_array)
                    comparison["max_difference"] = float(np.max(diff))
                    comparison["mean_difference"] = float(np.mean(diff))
                    comparison["outputs_equal"] = np.allclose(pt_array, tf_array, rtol=1e-5)

        except Exception as e:
            comparison["comparison_error"] = str(e)

        return comparison

    def _to_numpy(self, tensor: Any) -> np.ndarray | None:
        """Convert tensor to numpy array regardless of framework."""
        if tensor is None:
            return None

        try:
            # PyTorch tensor
            if hasattr(tensor, "detach"):
                return tensor.detach().cpu().numpy()
            # TensorFlow tensor
            elif hasattr(tensor, "numpy"):
                return tensor.numpy()
            # Already numpy
            elif isinstance(tensor, np.ndarray):
                return tensor
            # List or other sequence
            elif hasattr(tensor, "__iter__"):
                return np.array(tensor)
            else:
                return np.array([tensor])
        except Exception:
            return None

    def benchmark_performance(
        self,
        pytorch_func: Callable | None = None,
        tensorflow_func: Callable | None = None,
        inputs: dict[str, Any] = None,
        num_runs: int = 10,
        warmup_runs: int = 3,
    ) -> dict[str, Any]:
        """
        Benchmark performance of PyTorch vs TensorFlow implementations.

        Args:
            pytorch_func: PyTorch implementation
            tensorflow_func: TensorFlow implementation
            inputs: Input parameters
            num_runs: Number of timing runs
            warmup_runs: Number of warmup runs (not counted)

        Returns:
            Performance comparison results
        """
        results = {
            "pytorch": {"available": self.pytorch_available},
            "tensorflow": {"available": self.tensorflow_available},
        }

        # Benchmark PyTorch
        if pytorch_func and self.pytorch_available:
            results["pytorch"].update(self._benchmark_single(pytorch_func, inputs, num_runs, warmup_runs))

        # Benchmark TensorFlow
        if tensorflow_func and self.tensorflow_available:
            results["tensorflow"].update(self._benchmark_single(tensorflow_func, inputs, num_runs, warmup_runs))

        # Calculate relative performance
        if results["pytorch"].get("mean_time") and results["tensorflow"].get("mean_time"):
            pt_time = results["pytorch"]["mean_time"]
            tf_time = results["tensorflow"]["mean_time"]

            results["relative_performance"] = {
                "pytorch_faster": pt_time < tf_time,
                "speedup_ratio": max(pt_time, tf_time) / min(pt_time, tf_time),
                "faster_framework": "pytorch" if pt_time < tf_time else "tensorflow",
            }

        return results

    def _benchmark_single(
        self, func: Callable, inputs: dict[str, Any], num_runs: int, warmup_runs: int
    ) -> dict[str, Any]:
        """Benchmark a single implementation."""
        times = []

        try:
            # Warmup runs
            for _ in range(warmup_runs):
                if inputs:
                    func(**inputs)
                else:
                    func()

            # Actual timing runs
            for _ in range(num_runs):
                start_time = time.time()
                if inputs:
                    func(**inputs)
                else:
                    func()
                end_time = time.time()
                times.append(end_time - start_time)

            return {
                "success": True,
                "times": times,
                "mean_time": np.mean(times),
                "std_time": np.std(times),
                "min_time": np.min(times),
                "max_time": np.max(times),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def print_comparison_summary(self, comparison_name: str = None):
        """Print a formatted summary of comparison results."""
        if comparison_name:
            results = self.results.get(comparison_name)
            if not results:
                print(f"No results found for '{comparison_name}'")
                return
            self._print_single_comparison(results)
        else:
            for name, results in self.results.items():
                print(f"\n{'='*50}")
                print(f"Comparison: {name}")
                print("=" * 50)
                self._print_single_comparison(results)

    def _print_single_comparison(self, results: dict[str, Any]):
        """Print results for a single comparison."""
        # PyTorch results
        pt_results = results.get("pytorch", {})
        if pt_results.get("available"):
            if pt_results.get("success"):
                print(f"✅ PyTorch: {pt_results['execution_time']:.4f}s")
            else:
                print(f"❌ PyTorch: {pt_results.get('error', 'Unknown error')}")
        else:
            print("⚠️  PyTorch: Not available")

        # TensorFlow results
        tf_results = results.get("tensorflow", {})
        if tf_results.get("available"):
            if tf_results.get("success"):
                print(f"✅ TensorFlow: {tf_results['execution_time']:.4f}s")
            else:
                print(f"❌ TensorFlow: {tf_results.get('error', 'Unknown error')}")
        else:
            print("⚠️  TensorFlow: Not available")

        # Comparison results
        comparison = results.get("comparison", {})
        if comparison:
            print("\n📊 Comparison Results:")
            print(f"   Outputs equal: {comparison.get('outputs_equal', 'N/A')}")
            print(f"   Shape match: {comparison.get('shape_match', 'N/A')}")
            print(f"   Max difference: {comparison.get('max_difference', 'N/A')}")


def create_side_by_side_comparison(pytorch_code: str, tensorflow_code: str, title: str = "Framework Comparison") -> str:
    """
    Create a formatted side-by-side code comparison for display.

    Args:
        pytorch_code: PyTorch implementation code
        tensorflow_code: TensorFlow implementation code
        title: Title for the comparison

    Returns:
        Formatted string for display
    """
    lines_pt = pytorch_code.strip().split("\n")
    lines_tf = tensorflow_code.strip().split("\n")

    max_lines = max(len(lines_pt), len(lines_tf))
    max_width = 50

    # Pad shorter code block
    while len(lines_pt) < max_lines:
        lines_pt.append("")
    while len(lines_tf) < max_lines:
        lines_tf.append("")

    result = f"\n{title}\n{'='*100}\n"
    result += f"{'PyTorch':<{max_width}} | {'TensorFlow':<{max_width}}\n"
    result += f"{'-'*max_width} | {'-'*max_width}\n"

    for pt_line, tf_line in zip(lines_pt, lines_tf, strict=False):
        pt_display = pt_line[: max_width - 1] if len(pt_line) >= max_width else pt_line
        tf_display = tf_line[: max_width - 1] if len(tf_line) >= max_width else tf_line
        result += f"{pt_display:<{max_width}} | {tf_display:<{max_width}}\n"

    return result


# Convenience functions for common comparisons
def quick_tensor_comparison(pytorch_tensor, tensorflow_tensor) -> dict[str, Any]:
    """Quick comparison of two tensors from different frameworks."""
    comparator = FrameworkComparison()

    def pt_func():
        return pytorch_tensor

    def tf_func():
        return tensorflow_tensor

    return comparator.compare_implementations(pytorch_func=pt_func, tensorflow_func=tf_func, name="tensor_comparison")


def compare_model_outputs(pytorch_model, tensorflow_model, sample_input):
    """Compare outputs from equivalent PyTorch and TensorFlow models."""
    comparator = FrameworkComparison()

    def pt_func():
        if PYTORCH_AVAILABLE:
            pytorch_model.eval()
            with torch.no_grad():
                return pytorch_model(sample_input)
        return None

    def tf_func():
        if TENSORFLOW_AVAILABLE:
            return tensorflow_model(sample_input, training=False)
        return None

    return comparator.compare_implementations(pytorch_func=pt_func, tensorflow_func=tf_func, name="model_comparison")
