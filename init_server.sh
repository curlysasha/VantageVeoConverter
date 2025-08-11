#!/bin/bash
# VantageVeoConverter Server Initialization Script

echo "🚀 Initializing VantageVeoConverter on server..."

# Check if we're in virtual environment
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️ Not in virtual environment! Creating one..."
    python -m venv VantageVeoConverter
    source VantageVeoConverter/bin/activate
    echo "✅ Virtual environment activated"
fi

# Install system dependencies (Ubuntu/Debian)
echo "📦 Installing system dependencies..."
if command -v apt-get &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y espeak libasound2-dev libespeak-dev build-essential
    echo "✅ System dependencies installed"
fi

# Run automatic dependency installer
echo "🔧 Installing Python dependencies..."
python install_dependencies.py

if [ $? -ne 0 ]; then
    echo "❌ Dependency installation failed!"
    echo "💡 Try manual installation:"
    echo "   pip install numpy>=1.24,<2.3 scipy>=1.9.0"
    echo "   pip install -r requirements.txt"
    echo "   pip install --no-build-isolation aeneas>=1.7.3"
    exit 1
fi

# Setup binaries
echo "🔨 Setting up binary dependencies..."
python setup_binaries.py

# Test installation
echo "🧪 Testing installation..."
python test_binaries.py

if [ $? -eq 0 ]; then
    echo "🎉 Installation complete! Starting server..."
    python app_rife_compact.py
else
    echo "⚠️ Some tests failed, but you can try starting anyway:"
    echo "   python app_rife_compact.py"
fi