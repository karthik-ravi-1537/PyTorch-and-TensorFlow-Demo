# PyTorch and TensorFlow Demo

Comprehensive tutorial series comparing PyTorch and TensorFlow through practical, side-by-side implementations.

## What You'll Learn

- Framework fundamentals (tensors, computational graphs, debugging)
- Practical applications (NLP, computer vision, tabular data, time series)
- Production deployment (model serialization, inference, API development)
- Decision making (when to choose PyTorch vs TensorFlow)

## Tutorial Structure

### 1. Foundations
- NumPy essentials and data preparation
- Converting data between frameworks

### 2. Framework Fundamentals  
- Tensors, operations, and computational graphs
- Gradients, backpropagation, and debugging

### 3. Applications
- **NLP**: Text preprocessing, embeddings, classification, sequence modeling
- **Tabular Data**: Feature engineering, neural networks, classification/regression
- **Time Series**: Sequence preparation, forecasting, LSTM/GRU, attention

### 4. Production Bridge
- Model serialization and inference patterns
- API endpoints and deployment basics

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

## Getting Started

1. Start with `notebooks/01-foundations/numpy-essentials.ipynb`
2. Follow the numbered sequence within each section
3. Each notebook is self-contained with clear explanations

## Project Structure

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
└── data/                   # Sample datasets
```

## Framework Comparison

| Aspect | PyTorch | TensorFlow |
|--------|---------|------------|
| **Learning Curve** | Easier | Steeper |
| **Research** | Preferred in academia | Growing adoption |
| **Production** | Requires additional tools | Built-in deployment |
| **Debugging** | Native Python debugging | More challenging |
| **Mobile/Web** | Limited options | Excellent (TF Lite, TF.js) |

## Contributing

### Development Setup

```bash
# Clone and install with dev dependencies
uv sync --dev
source .venv/bin/activate

# Install development tools
uv pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install

# Run formatting and linting
pre-commit run --all-files
```

### Development Tools

- **ruff**: Fast Python linter and formatter
- **black**: Code formatter
- **pre-commit**: Git hooks for code quality
- **pytest**: Testing framework