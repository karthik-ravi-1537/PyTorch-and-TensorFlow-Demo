#!/usr/bin/env python3
"""
Setup verification for PyTorch and TensorFlow Demo project.
"""
import sys
import os
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        import torch
        import tensorflow as tf
        print("✅ Core ML frameworks imported successfully")
        
        import numpy
        import pandas
        import matplotlib
        import seaborn
        import plotly
        print("✅ Data science packages imported successfully")
        
        import jupyter
        import jupyterlab
        print("✅ Jupyter packages imported successfully")
        
        import nltk
        import transformers
        print("✅ NLP packages imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_framework_versions():
    """Test framework versions and basic functionality."""
    print("\n⚙️  Testing framework versions...")
    
    try:
        import torch
        import tensorflow as tf
        
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ TensorFlow version: {tf.__version__}")
        
        # Test basic tensor operations
        pytorch_tensor = torch.tensor([1, 2, 3])
        tf_tensor = tf.constant([1, 2, 3])
        
        print("✅ Basic tensor operations working")
        return True
        
    except Exception as e:
        print(f"❌ Framework test error: {e}")
        return False

def test_jupyter_environment():
    """Test Jupyter environment setup."""
    print("\n📓 Testing Jupyter environment...")
    
    try:
        import jupyter
        import jupyterlab
        import ipywidgets
        
        print("✅ Jupyter environment components available")
        return True
        
    except Exception as e:
        print(f"❌ Jupyter test error: {e}")
        return False

def test_project_structure():
    """Test that all expected files and directories exist."""
    print("\n📁 Testing project structure...")
    
    expected_files = [
        "README.md",
        "requirements.txt",
        "environment.yml",
        "pyproject.toml",
        ".python-version",
        ".pre-commit-config.yaml",
        ".gitignore",
        "notebooks/01-foundations/numpy-essentials.ipynb",
        "notebooks/02-framework-fundamentals/tensors-and-operations.ipynb",
        "src/pytorch_examples/tabular/mlp_classifier.py",
        "src/tensorflow_examples/tabular/mlp_classifier.py"
    ]
    
    missing_files = []
    for file_path in expected_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            print(f"✅ {file_path}")
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✅ All expected files found")
    return True

def main():
    """Run all tests."""
    print("🚀 PyTorch and TensorFlow Demo - Setup Test")
    print("=" * 50)
    
    tests = [
        ("Project Structure", test_project_structure),
        ("Imports", test_imports),
        ("Framework Versions", test_framework_versions),
        ("Jupyter Environment", test_jupyter_environment)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Your setup is ready.")
        print("\n💡 Next steps:")
        print("1. Launch Jupyter Lab: jupyter lab")
        print("2. Start with: notebooks/01-foundations/numpy-essentials.ipynb")
        print("3. Follow the tutorial sequence")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
