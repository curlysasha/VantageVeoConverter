# VantageVeoConverter Installation Guide

## Quick Setup

### Method 1: Automatic Installation (Recommended)
```bash
# Create virtual environment
python -m venv VantageVeoConverter
source VantageVeoConverter/bin/activate  # Linux/Mac
# OR
VantageVeoConverter\Scripts\activate     # Windows

# Run automatic installer
python install_dependencies.py
```

### Method 2: Manual Installation
```bash
# Create virtual environment
python -m venv VantageVeoConverter
source VantageVeoConverter/bin/activate

# Install in correct order
pip install numpy>=1.24,<2.3 scipy>=1.9.0

# Install main dependencies
pip install -r requirements.txt

# Install aeneas separately (choose one that works):
pip install --no-build-isolation aeneas>=1.7.3
# OR if above fails:
pip install --no-deps aeneas>=1.7.3
```

### Method 3: Using uv (Fast Package Manager)
```bash
# Create virtual environment
uv venv VantageVeoConverter
source VantageVeoConverter/bin/activate

# Install core first
uv pip install numpy>=1.24,<2.3 scipy>=1.9.0

# Install main dependencies
uv pip install -r requirements.txt

# Install aeneas
uv pip install --no-build-isolation aeneas>=1.7.3
```

## System Dependencies

### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install espeak libasound2-dev libespeak-dev
```

### CentOS/RHEL
```bash
sudo yum install espeak-devel alsa-lib-devel
```

### Conda Alternative
If pip installation fails:
```bash
conda install -c conda-forge aeneas
```

## Binary Dependencies

The application needs these binaries in the `bin/` directory:
- `ffmpeg` - Video processing
- `ffprobe` - Video analysis  
- `mp4fpsmod` - VFR timecode application

### Automatic Binary Setup
```bash
python setup_binaries.py
```

### Manual Binary Setup
1. Download FFmpeg: https://johnvansickle.com/ffmpeg/releases/
2. Build mp4fpsmod: https://github.com/nu774/mp4fpsmod
3. Place binaries in `bin/` directory

## Troubleshooting

### Aeneas Installation Issues
```bash
# Force numpy reinstall
pip install --force-reinstall numpy>=1.24,<2.3

# Try aeneas again
pip install aeneas>=1.7.3
```

### GPU Support
For CUDA acceleration:
```bash
# Install CUDA PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Test Installation
```bash
python test_binaries.py  # Test binary detection
python -c "import aeneas; print('Aeneas OK')"  # Test aeneas
```

## Running the Application
```bash
python app_rife_compact.py
```

The application will be available at `http://localhost:7860`