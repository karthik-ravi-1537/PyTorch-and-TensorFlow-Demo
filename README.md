# PyTorch and TensorFlow Demo

A comprehensive tutorial series comparing PyTorch and TensorFlow through practical, side-by-side implementations. Learn both frameworks through hands-on examples in NLP, computer vision, tabular data, and time series applications.

## 🎯 **What You'll Learn**

- **Framework Fundamentals**: Core concepts, tensors, computational graphs, and debugging
- **Practical Applications**: Real-world examples in NLP, tabular data, and time series
- **Production Deployment**: Model serialization, inference patterns, and API development
- **Decision Making**: When to choose PyTorch vs TensorFlow for your projects

## 📚 **Tutorial Structure**

### **1. Foundations**
- **NumPy Essentials**: Arrays, operations, broadcasting, and framework integration
- **Pandas for ML**: DataFrames, feature engineering, data preparation
- **Data Preparation**: Converting data between NumPy/Pandas and ML frameworks

### **2. Framework Fundamentals**
- **Tensors and Operations**: Core data structures and manipulations
- **Computational Graphs**: Dynamic vs static execution models
- **Gradients and Backpropagation**: Automatic differentiation patterns
- **Debugging Strategies**: Framework-specific troubleshooting

### **3. NLP Applications**
- **Text Preprocessing**: Tokenization and cleaning approaches
- **Embeddings**: Word representations and usage patterns
- **Text Classification**: Neural networks for text analysis
- **Sequence Modeling**: RNNs, LSTMs, and attention mechanisms

### **4. Tabular Data**
- **Feature Engineering**: Preprocessing for neural networks
- **Neural Networks**: Deep learning for structured data
- **Classification**: Binary and multi-class prediction
- **Regression**: Continuous target variable modeling

### **5. Time Series**
- **Sequence Preparation**: Time series preprocessing techniques
- **Forecasting Models**: Prediction and trend analysis
- **LSTM/GRU Comparison**: Recurrent architecture patterns
- **Attention Mechanisms**: Modern approaches to sequence modeling

### **6. Production Bridge**
- **Model Serialization**: Saving and loading trained models
- **Inference Patterns**: Batch and real-time prediction
- **API Endpoints**: Basic model serving with web frameworks
- **Deployment Basics**: Containerization and MLOps introduction

## Installation

### Prerequisites
- [Homebrew](https://brew.sh/) for installing uv
- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

### Method 1: uv

```bash
brew install uv

git clone https://github.com/karthik-ravi-1537/PyTorch-and-TensorFlow-Demo.git
cd PyTorch-and-TensorFlow-Demo

uv sync

source .venv/bin/activate
```

### Method 2: conda

```bash
git clone https://github.com/karthik-ravi-1537/PyTorch-and-TensorFlow-Demo.git
cd PyTorch-and-TensorFlow-Demo

conda env create -f environment.yml
conda activate pytorch-and-tensorflow-demo
```

### Verification
```bash
python test_setup.py
```

### **Getting Started**
1. Start with `notebooks/01-foundations/numpy-essentials.ipynb`
2. Follow the numbered sequence within each section
3. Each notebook is self-contained with clear explanations
4. Focus on understanding concepts and framework differences

## 📖 **How to Use This Tutorial**

### **Learning Paths**
- **Beginners**: Start from foundations and work through sequentially
- **Experienced Developers**: Jump to framework fundamentals or specific domains
- **Framework Switchers**: Focus on comparison sections and decision guides

### **Notebook Features**
- **Side-by-side comparisons** of PyTorch and TensorFlow implementations
- **Detailed explanations** of framework differences and trade-offs
- **Performance benchmarks** and memory usage comparisons
- **"When to use" recommendations** for different scenarios
- **Production-ready code examples** with best practices

## 🛠️ **Project Structure**

```
PyTorch-and-TensorFlow-Demo/
├── notebooks/              # Interactive Jupyter tutorials
│   ├── 01-foundations/     # NumPy, Pandas, data preparation
│   ├── 02-framework-fundamentals/  # Core PyTorch vs TensorFlow
│   ├── 03-nlp-applications/        # Natural language processing
│   ├── 04-tabular-data/           # Structured data applications
│   ├── 05-time-series/            # Temporal data modeling
│   └── 06-production-bridge/      # Deployment and production
├── src/                    # Production-ready Python modules
│   ├── pytorch_examples/   # Clean PyTorch implementations
│   ├── tensorflow_examples/# Clean TensorFlow implementations
│   └── utils/             # Shared utilities and tools
├── data/                  # Sample datasets and examples
├── reference/             # Quick reference guides and comparisons
└── docs/                  # Contributing guidelines
```

## 🎓 **Learning Approach**

This tutorial is designed for **hands-on learning**:
- **Complete, runnable examples** in every notebook
- **No coding required** during tutorials - focus on understanding
- **Clear explanations** of framework trade-offs and design decisions
- **Practical guidance** for real-world applications
- **Progressive complexity** from basic concepts to production deployment

## 🔄 **Framework Philosophy**

Rather than favoring one framework, this tutorial:
- **Presents both frameworks objectively** with their strengths and use cases
- **Provides practical decision-making guidance** based on project requirements
- **Shows equivalent implementations side-by-side** for direct comparison
- **Explains the "why"** behind framework differences and design choices

## 📊 **Framework Comparison**

| Aspect | PyTorch | TensorFlow |
|--------|---------|------------|
| **Learning Curve** | Easier for beginners | Steeper, but improving |
| **Research** | Preferred in academia | Growing adoption |
| **Production** | Requires additional tools | Built-in deployment features |
| **Debugging** | Native Python debugging | More challenging |
| **Mobile/Web** | Limited options | Excellent (TF Lite, TF.js) |

**👉 See [Framework Comparison Guide](reference/comparison-matrix.md) for detailed analysis**

## 🚀 **Next Steps**

After completing this tutorial:
- **Choose the right framework** for your specific projects
- **Implement production solutions** using framework-specific best practices
- **Understand deployment patterns** and MLOps considerations
- **Make informed decisions** about framework selection for teams

## 🤝 **Contributing**

We welcome contributions! Please see our [Contributing Guide](docs/contributing.md) for:
- Adding new examples or improving existing content
- Reporting issues or suggesting enhancements
- Keeping content up-to-date with framework changes
- Sharing your learning experiences and feedback

## 📞 **Support**

- **Questions**: Open an issue or start a discussion
- **Framework Decisions**: Use our [Decision Guide](reference/when-to-use-what.md)
- **Contributing**: See our [Contributing Guide](docs/contributing.md)

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- **PyTorch and TensorFlow communities** for excellent documentation and tools
- **Open source contributors** who make machine learning accessible
- **Educators and learners** who inspire better educational resources

---

**Ready to master both frameworks?** Start with `notebooks/01-foundations/numpy-essentials.ipynb` and begin your journey! 🚀