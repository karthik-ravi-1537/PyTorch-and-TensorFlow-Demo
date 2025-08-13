"""
Visualization utilities for comparing PyTorch and TensorFlow implementations.
Provides plotting functions for performance comparisons, training curves, and framework differences.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings

# Set style
plt.style.use('default')
sns.set_palette("husl")

class FrameworkVisualizer:
    """Main class for creating visualizations comparing PyTorch and TensorFlow."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        self.colors = {
            'pytorch': '#EE4C2C',  # PyTorch orange
            'tensorflow': '#FF6F00',  # TensorFlow orange
            'comparison': '#1f77b4'  # Blue for comparisons
        }
    
    def plot_performance_comparison(
        self,
        benchmark_results: Dict[str, Any],
        title: str = "Performance Comparison",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot performance comparison between PyTorch and TensorFlow.
        
        Args:
            benchmark_results: Results from FrameworkComparison.benchmark_performance()
            title: Plot title
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # Extract data
        frameworks = []
        mean_times = []
        std_times = []
        
        for framework in ['pytorch', 'tensorflow']:
            if (framework in benchmark_results and 
                benchmark_results[framework].get('success')):
                frameworks.append(framework.capitalize())
                mean_times.append(benchmark_results[framework]['mean_time'])
                std_times.append(benchmark_results[framework]['std_time'])
        
        if not frameworks:
            ax1.text(0.5, 0.5, 'No successful benchmarks', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax2.text(0.5, 0.5, 'No data available', 
                    ha='center', va='center', transform=ax2.transAxes)
            return fig
        
        # Bar plot of mean execution times
        colors = [self.colors[fw.lower()] for fw in frameworks]
        bars = ax1.bar(frameworks, mean_times, yerr=std_times, 
                      capsize=5, color=colors, alpha=0.7)
        ax1.set_ylabel('Execution Time (seconds)')
        ax1.set_title('Mean Execution Time')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, mean_time, std_time in zip(bars, mean_times, std_times):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + std_time,
                    f'{mean_time:.4f}s', ha='center', va='bottom')
        
        # Box plot of all timing runs if available
        timing_data = []
        timing_labels = []
        
        for framework in ['pytorch', 'tensorflow']:
            if (framework in benchmark_results and 
                benchmark_results[framework].get('success') and
                'times' in benchmark_results[framework]):
                timing_data.append(benchmark_results[framework]['times'])
                timing_labels.append(framework.capitalize())
        
        if timing_data:
            bp = ax2.boxplot(timing_data, labels=timing_labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax2.set_ylabel('Execution Time (seconds)')
            ax2.set_title('Execution Time Distribution')
            ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_training_curves(
        self,
        pytorch_history: Optional[Dict[str, List[float]]] = None,
        tensorflow_history: Optional[Dict[str, List[float]]] = None,
        metrics: List[str] = ['loss', 'accuracy'],
        title: str = "Training Curves Comparison",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot training curves for both frameworks.
        
        Args:
            pytorch_history: PyTorch training history
            tensorflow_history: TensorFlow training history
            metrics: List of metrics to plot
            title: Plot title
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 6))
        
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Plot PyTorch history
            if pytorch_history and metric in pytorch_history:
                epochs = range(1, len(pytorch_history[metric]) + 1)
                ax.plot(epochs, pytorch_history[metric], 
                       color=self.colors['pytorch'], linewidth=2, 
                       label=f'PyTorch {metric}', marker='o', markersize=4)
            
            # Plot TensorFlow history
            if tensorflow_history and metric in tensorflow_history:
                epochs = range(1, len(tensorflow_history[metric]) + 1)
                ax.plot(epochs, tensorflow_history[metric], 
                       color=self.colors['tensorflow'], linewidth=2, 
                       label=f'TensorFlow {metric}', marker='s', markersize=4)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f'{metric.capitalize()} Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_memory_usage(
        self,
        pytorch_memory: Optional[List[float]] = None,
        tensorflow_memory: Optional[List[float]] = None,
        time_points: Optional[List[float]] = None,
        title: str = "Memory Usage Comparison",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot memory usage over time for both frameworks.
        
        Args:
            pytorch_memory: PyTorch memory usage data
            tensorflow_memory: TensorFlow memory usage data
            time_points: Time points for measurements
            title: Plot title
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if time_points is None:
            if pytorch_memory:
                time_points = range(len(pytorch_memory))
            elif tensorflow_memory:
                time_points = range(len(tensorflow_memory))
            else:
                time_points = []
        
        if pytorch_memory:
            ax.plot(time_points, pytorch_memory, 
                   color=self.colors['pytorch'], linewidth=2,
                   label='PyTorch', marker='o', markersize=4)
        
        if tensorflow_memory:
            ax.plot(time_points, tensorflow_memory, 
                   color=self.colors['tensorflow'], linewidth=2,
                   label='TensorFlow', marker='s', markersize=4)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Memory Usage (MB)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_accuracy_comparison(
        self,
        comparison_results: Dict[str, Any],
        title: str = "Output Accuracy Comparison",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot accuracy comparison between framework outputs.
        
        Args:
            comparison_results: Results from FrameworkComparison
            title: Plot title
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # Extract comparison data
        comparison = comparison_results.get('comparison', {})
        
        if not comparison:
            ax1.text(0.5, 0.5, 'No comparison data available', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax2.text(0.5, 0.5, 'No comparison data available', 
                    ha='center', va='center', transform=ax2.transAxes)
            return fig
        
        # Accuracy metrics
        metrics = ['outputs_equal', 'shape_match', 'dtype_match']
        values = [comparison.get(metric, False) for metric in metrics]
        colors_list = ['green' if v else 'red' for v in values]
        
        ax1.bar(metrics, [1 if v else 0 for v in values], color=colors_list, alpha=0.7)
        ax1.set_ylabel('Match (1=True, 0=False)')
        ax1.set_title('Comparison Metrics')
        ax1.set_ylim(0, 1.2)
        
        # Add text labels
        for i, (metric, value) in enumerate(zip(metrics, values)):
            ax1.text(i, 0.5, str(value), ha='center', va='center', 
                    fontweight='bold', color='white')
        
        # Difference metrics
        max_diff = comparison.get('max_difference')
        mean_diff = comparison.get('mean_difference')
        
        if max_diff is not None and mean_diff is not None:
            diff_metrics = ['Max Difference', 'Mean Difference']
            diff_values = [max_diff, mean_diff]
            
            ax2.bar(diff_metrics, diff_values, color=self.colors['comparison'], alpha=0.7)
            ax2.set_ylabel('Absolute Difference')
            ax2.set_title('Numerical Differences')
            ax2.set_yscale('log')
            
            # Add value labels
            for i, value in enumerate(diff_values):
                ax2.text(i, value, f'{value:.2e}', ha='center', va='bottom')
        else:
            ax2.text(0.5, 0.5, 'No numerical differences available', 
                    ha='center', va='center', transform=ax2.transAxes)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_framework_comparison_dashboard(
        self,
        comparison_results: Dict[str, Any],
        benchmark_results: Optional[Dict[str, Any]] = None,
        title: str = "Framework Comparison Dashboard",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a comprehensive dashboard comparing frameworks.
        
        Args:
            comparison_results: Results from framework comparison
            benchmark_results: Optional benchmark results
            title: Dashboard title
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        if benchmark_results:
            fig = plt.figure(figsize=(16, 12))
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        else:
            fig = plt.figure(figsize=(12, 8))
            gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Execution status
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_execution_status(ax1, comparison_results)
        
        # Accuracy comparison
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_accuracy_metrics(ax2, comparison_results)
        
        # Numerical differences
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_numerical_differences(ax3, comparison_results)
        
        # Framework info
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_framework_info(ax4, comparison_results)
        
        # Performance comparison (if available)
        if benchmark_results:
            ax5 = fig.add_subplot(gs[2, :])
            self._plot_performance_summary(ax5, benchmark_results)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _plot_execution_status(self, ax, results):
        """Plot execution status for both frameworks."""
        frameworks = ['PyTorch', 'TensorFlow']
        statuses = []
        colors = []
        
        for fw in ['pytorch', 'tensorflow']:
            if fw in results:
                if results[fw].get('success'):
                    statuses.append('Success')
                    colors.append('green')
                else:
                    statuses.append('Failed')
                    colors.append('red')
            else:
                statuses.append('N/A')
                colors.append('gray')
        
        bars = ax.bar(frameworks, [1]*len(frameworks), color=colors, alpha=0.7)
        ax.set_ylabel('Status')
        ax.set_title('Execution Status')
        ax.set_ylim(0, 1.2)
        
        for bar, status in zip(bars, statuses):
            ax.text(bar.get_x() + bar.get_width()/2., 0.5, status,
                   ha='center', va='center', fontweight='bold', color='white')
    
    def _plot_accuracy_metrics(self, ax, results):
        """Plot accuracy metrics."""
        comparison = results.get('comparison', {})
        metrics = ['outputs_equal', 'shape_match', 'dtype_match']
        values = [comparison.get(metric, False) for metric in metrics]
        colors = ['green' if v else 'red' for v in values]
        
        ax.bar(metrics, [1 if v else 0 for v in values], color=colors, alpha=0.7)
        ax.set_ylabel('Match')
        ax.set_title('Accuracy Metrics')
        ax.set_ylim(0, 1.2)
        
        for i, (metric, value) in enumerate(zip(metrics, values)):
            ax.text(i, 0.5, str(value), ha='center', va='center',
                   fontweight='bold', color='white')
    
    def _plot_numerical_differences(self, ax, results):
        """Plot numerical differences."""
        comparison = results.get('comparison', {})
        max_diff = comparison.get('max_difference')
        mean_diff = comparison.get('mean_difference')
        
        if max_diff is not None and mean_diff is not None:
            metrics = ['Max Diff', 'Mean Diff']
            values = [max_diff, mean_diff]
            
            ax.bar(metrics, values, color=self.colors['comparison'], alpha=0.7)
            ax.set_ylabel('Absolute Difference')
            ax.set_title('Numerical Differences')
            ax.set_yscale('log')
            
            for i, value in enumerate(values):
                ax.text(i, value, f'{value:.2e}', ha='center', va='bottom')
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                   transform=ax.transAxes)
            ax.set_title('Numerical Differences')
    
    def _plot_framework_info(self, ax, results):
        """Plot framework availability and timing info."""
        info_text = []
        
        for fw in ['pytorch', 'tensorflow']:
            if fw in results:
                available = results[fw].get('available', False)
                exec_time = results[fw].get('execution_time', 0)
                info_text.append(f"{fw.capitalize()}:")
                info_text.append(f"  Available: {available}")
                if available and exec_time > 0:
                    info_text.append(f"  Time: {exec_time:.4f}s")
                info_text.append("")
        
        ax.text(0.1, 0.9, '\n'.join(info_text), transform=ax.transAxes,
               verticalalignment='top', fontfamily='monospace')
        ax.set_title('Framework Info')
        ax.axis('off')
    
    def _plot_performance_summary(self, ax, benchmark_results):
        """Plot performance summary."""
        frameworks = []
        mean_times = []
        
        for fw in ['pytorch', 'tensorflow']:
            if (fw in benchmark_results and 
                benchmark_results[fw].get('success')):
                frameworks.append(fw.capitalize())
                mean_times.append(benchmark_results[fw]['mean_time'])
        
        if frameworks:
            colors = [self.colors[fw.lower()] for fw in frameworks]
            bars = ax.bar(frameworks, mean_times, color=colors, alpha=0.7)
            ax.set_ylabel('Mean Execution Time (s)')
            ax.set_title('Performance Summary')
            
            for bar, time in zip(bars, mean_times):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                       f'{time:.4f}s', ha='center', va='bottom')
        else:
            ax.text(0.5, 0.5, 'No benchmark data', ha='center', va='center',
                   transform=ax.transAxes)
            ax.set_title('Performance Summary')


def plot_tensor_comparison(pytorch_tensor, tensorflow_tensor, title="Tensor Comparison"):
    """Quick visualization of tensor differences."""
    visualizer = FrameworkVisualizer()
    
    # Convert to numpy
    try:
        pt_array = pytorch_tensor.detach().cpu().numpy() if hasattr(pytorch_tensor, 'detach') else pytorch_tensor
        tf_array = tensorflow_tensor.numpy() if hasattr(tensorflow_tensor, 'numpy') else tensorflow_tensor
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # PyTorch tensor
        if pt_array.ndim == 2:
            im1 = axes[0].imshow(pt_array, cmap='viridis')
            plt.colorbar(im1, ax=axes[0])
        else:
            axes[0].plot(pt_array.flatten())
        axes[0].set_title('PyTorch')
        
        # TensorFlow tensor
        if tf_array.ndim == 2:
            im2 = axes[1].imshow(tf_array, cmap='viridis')
            plt.colorbar(im2, ax=axes[1])
        else:
            axes[1].plot(tf_array.flatten())
        axes[1].set_title('TensorFlow')
        
        # Difference
        diff = np.abs(pt_array - tf_array)
        if diff.ndim == 2:
            im3 = axes[2].imshow(diff, cmap='Reds')
            plt.colorbar(im3, ax=axes[2])
        else:
            axes[2].plot(diff.flatten())
        axes[2].set_title(f'Absolute Difference\nMax: {np.max(diff):.2e}')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        print(f"Error creating tensor comparison plot: {e}")
        return None