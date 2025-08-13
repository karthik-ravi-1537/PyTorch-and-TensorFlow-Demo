# 🤝 Contributing to PyTorch vs TensorFlow Demo

Thank you for your interest in contributing! This guide will help you get started with contributing to this educational project.

## 🎯 Project Goals

This project aims to provide:
- **Clear, practical comparisons** between PyTorch and TensorFlow
- **Hands-on learning experiences** through Jupyter notebooks
- **Production-ready code examples** for both frameworks
- **Comprehensive documentation** for learners at all levels

## 🚀 Getting Started

### 1. Fork and Clone
```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/karthik-ravi-1537/pytorch-tensorflow-demo.git
cd pytorch-tensorflow-demo

# Add upstream remote
git remote add upstream https://github.com/ORIGINAL_OWNER/pytorch-tensorflow-demo.git
```

### 2. Set Up Development Environment
```bash
# Create development environment
conda env create -f environment.yml
conda activate ml-frameworks-tutorial

# Install development dependencies
pip install black flake8 mypy pytest pre-commit

# Set up pre-commit hooks (optional but recommended)
pre-commit install

# Verify setup
python test_imports.py
```

### 3. Create a Branch
```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/issue-description
```

## 📝 Types of Contributions

### 🐛 Bug Reports
**Before submitting:**
- Check existing issues
- Run `python test_imports.py`
- Include system information

**Include in your report:**
- Clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)
- Error messages (full traceback)

### 💡 Feature Requests
**Good feature requests:**
- Align with project goals
- Include clear use cases
- Consider both PyTorch and TensorFlow
- Provide implementation suggestions

### 📚 Documentation Improvements
**Areas for improvement:**
- Clarifying existing explanations
- Adding missing examples
- Fixing typos or formatting
- Improving code comments
- Adding troubleshooting tips

### 🔧 Code Contributions
**Types of code contributions:**
- New tutorial notebooks
- Bug fixes in existing code
- Performance improvements
- Additional utility functions
- Test improvements

## 📋 Contribution Guidelines

### Notebook Contributions

#### Structure Requirements
```python
# Every notebook should start with:
{
 "cell_type": "markdown",
 "metadata": {},
 "source": [
  "# Notebook Title\n",
  "\n",
  "**Learning Objectives:**\n",
  "- Clear, specific learning goals\n",
  "- Measurable outcomes\n",
  "\n",
  "**Prerequisites:** List required knowledge\n",
  "\n",
  "**Estimated Time:** X minutes"
 ]
}
```

#### Content Guidelines
- **Side-by-side comparisons**: Always show both PyTorch and TensorFlow
- **Practical examples**: Use realistic data and scenarios
- **Clear explanations**: Explain the "why" not just the "how"
- **Progressive complexity**: Start simple, build up gradually
- **Working code**: All cells must execute without errors

#### Code Style in Notebooks
```python
# Good: Clear, commented code
import torch
import torch.nn as nn

# Create a simple neural network
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Bad: Uncommented, unclear code
net = nn.Sequential(nn.Linear(10,5), nn.ReLU(), nn.Linear(5,1))
```

### Source Code Contributions

#### Code Style
We use **Black** for formatting and **flake8** for linting:
```bash
# Format code
black src/ notebooks/

# Check linting
flake8 src/

# Type checking (optional)
mypy src/
```

#### Documentation Standards
```python
def example_function(param1: str, param2: int = 10) -> bool:
    """
    Brief description of what the function does.
    
    Args:
        param1: Description of parameter 1
        param2: Description of parameter 2 (default: 10)
    
    Returns:
        Description of return value
    
    Example:
        >>> result = example_function("test", 5)
        >>> print(result)
        True
    """
    # Implementation here
    return True
```

#### Testing Requirements
- All new functions must have tests
- Tests should cover both PyTorch and TensorFlow paths
- Use descriptive test names
- Include edge cases

```python
def test_data_loading_pytorch():
    """Test that PyTorch data loading works correctly."""
    data = get_tutorial_tabular_data(num_samples=10)
    assert data['X'].shape == (10, 10)
    assert data['y'].shape == (10,)
```

## 🔄 Development Workflow

### 1. Before You Start
```bash
# Sync with upstream
git fetch upstream
git checkout main
git merge upstream/main

# Create feature branch
git checkout -b feature/your-feature
```

### 2. During Development
```bash
# Make changes
# Test your changes
python test_imports.py

# Run specific notebook to test
jupyter lab notebooks/your-notebook.ipynb

# Format code
black src/

# Commit changes
git add .
git commit -m "feat: add new comparison notebook for CNNs"
```

### 3. Before Submitting
```bash
# Run full test suite
python test_imports.py

# Check all notebooks work
# (Manual testing for now, automated testing coming soon)

# Update documentation if needed
# Add entry to CHANGELOG.md if significant change
```

### 4. Submit Pull Request
- **Clear title**: Describe what the PR does
- **Detailed description**: Explain the changes and why
- **Link issues**: Reference any related issues
- **Screenshots**: For UI/notebook changes
- **Testing notes**: How you tested the changes

## 📐 Standards and Best Practices

### Notebook Standards
- **Consistent structure**: Follow existing notebook patterns
- **Clear outputs**: Include example outputs in committed notebooks
- **No sensitive data**: Never commit API keys, personal data, etc.
- **Reasonable execution time**: Keep examples under 5 minutes per notebook
- **Cross-platform compatibility**: Test on different operating systems

### Code Standards
- **Framework agnostic utilities**: Write reusable code when possible
- **Error handling**: Include appropriate try/catch blocks
- **Performance considerations**: Avoid unnecessary computations
- **Memory management**: Clean up large objects when done
- **Logging**: Use appropriate logging levels

### Documentation Standards
- **Clear language**: Write for learners, not experts
- **Complete examples**: Include full, runnable code
- **Visual aids**: Use diagrams and plots when helpful
- **Up-to-date**: Keep documentation current with code changes

## 🧪 Testing Guidelines

### Manual Testing Checklist
- [ ] All notebook cells execute without errors
- [ ] Outputs are reasonable and expected
- [ ] Both PyTorch and TensorFlow examples work
- [ ] Code follows style guidelines
- [ ] Documentation is clear and accurate

### Automated Testing (Future)
We're working on:
- Automated notebook execution testing
- Code quality checks
- Performance regression testing
- Documentation completeness validation

## 📚 Resources for Contributors

### Learning Resources
- **PyTorch Documentation**: https://pytorch.org/docs/
- **TensorFlow Documentation**: https://www.tensorflow.org/guide
- **Jupyter Best Practices**: https://jupyter.readthedocs.io/
- **Python Style Guide**: https://pep8.org/

### Project-Specific Resources
- **Architecture Overview**: See `docs/architecture.md` (if exists)
- **Design Decisions**: Check issue discussions
- **Roadmap**: See project milestones and issues

## 🏷️ Issue Labels

We use these labels to organize contributions:
- **good first issue**: Great for new contributors
- **help wanted**: We need community help
- **bug**: Something isn't working
- **enhancement**: New feature or improvement
- **documentation**: Documentation improvements
- **notebook**: Jupyter notebook related
- **pytorch**: PyTorch specific
- **tensorflow**: TensorFlow specific

## 🎉 Recognition

Contributors are recognized in:
- **README.md**: Contributors section
- **Release notes**: Major contributions highlighted
- **Hall of Fame**: Top contributors featured

## 📞 Getting Help

### Questions About Contributing
- **GitHub Discussions**: For general questions
- **Issues**: For specific problems
- **Email**: [maintainer-email] for sensitive topics

### Code Review Process
1. **Automated checks**: CI/CD runs tests
2. **Maintainer review**: Code quality and alignment
3. **Community feedback**: Other contributors may comment
4. **Approval**: Maintainer approves and merges

## 📋 Contribution Checklist

Before submitting a PR:
- [ ] Code follows style guidelines (Black, flake8)
- [ ] All tests pass (`python test_imports.py`)
- [ ] Documentation is updated
- [ ] Commit messages are clear
- [ ] PR description explains changes
- [ ] No sensitive information included
- [ ] Both frameworks tested (if applicable)

## 🔄 Maintenance Guidelines

### Keeping Dependencies Current
```bash
# Check for outdated packages
pip list --outdated

# Update requirements files
pip-compile requirements.in  # if using pip-tools

# Test with new versions
python test_imports.py
```

### Framework Version Updates
When PyTorch or TensorFlow releases new versions:
1. **Test compatibility** with existing notebooks
2. **Update version requirements** in requirements.txt
3. **Document breaking changes** in CHANGELOG.md
4. **Update installation guides** if needed

### Content Maintenance
- **Regular review**: Check notebooks still work with new versions
- **Performance updates**: Optimize slow examples
- **Content freshness**: Update examples with current best practices
- **Link checking**: Ensure external links still work

---

Thank you for contributing to make machine learning education better! 🚀

*Last updated: December 2024*