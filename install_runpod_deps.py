#!/usr/bin/env python3
"""
RunPod-specific dependency installer for VantageVeoConverter
Optimized for Docker container environment with proper error handling
"""
import subprocess
import sys
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_pip_command(args, description, required=True):
    """Run pip command with enhanced error handling"""
    logger.info(f"📦 {description}...")
    
    cmd = [sys.executable, "-m", "pip", "install"] + args
    logger.info(f"   Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=600)
        logger.info(f"✅ {description} - SUCCESS")
        if result.stdout:
            logger.debug(f"   Output: {result.stdout[:200]}...")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} - FAILED")
        logger.error(f"   Error: {e.stderr}")
        if not required:
            logger.warning(f"   Continuing without {description} (not required)")
            return False
        return False
    except subprocess.TimeoutExpired:
        logger.error(f"❌ {description} - TIMEOUT")
        return False

def check_system_dependencies():
    """Check if system dependencies are available"""
    logger.info("🔍 Checking system dependencies...")
    
    deps = {
        "ffmpeg": "FFmpeg binary",
        "espeak": "eSpeak for aeneas",
        "gcc": "GCC compiler"
    }
    
    missing = []
    for dep, desc in deps.items():
        try:
            result = subprocess.run(["which", dep], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"✅ {desc} found at {result.stdout.strip()}")
            else:
                missing.append(dep)
                logger.warning(f"⚠️ {desc} not found")
        except FileNotFoundError:
            missing.append(dep)
            logger.warning(f"⚠️ {desc} not found")
    
    return missing

def install_core_dependencies():
    """Install core Python dependencies"""
    logger.info("1️⃣ Installing core dependencies...")
    
    # Critical: Install correct setuptools version first
    if not run_pip_command(["setuptools==59.5.0"], "setuptools (aeneas compatible)", required=True):
        return False
    
    # Install numpy/scipy with specific versions
    if not run_pip_command([
        "numpy>=1.24,<2.3",
        "scipy>=1.9.0"
    ], "Core numerical libraries", required=True):
        return False
    
    return True

def install_pytorch():
    """Install PyTorch with CUDA support"""
    logger.info("2️⃣ Installing PyTorch...")
    
    # Use CUDA 11.8 index for compatibility with RunPod
    pytorch_args = [
        "torch>=1.12.0",
        "torchvision>=0.13.0", 
        "torchaudio>=0.12.0",
        "--index-url", "https://download.pytorch.org/whl/cu118"
    ]
    
    return run_pip_command(pytorch_args, "PyTorch with CUDA support", required=True)

def install_main_dependencies():
    """Install main application dependencies"""
    logger.info("3️⃣ Installing main dependencies...")
    
    deps = [
        "runpod>=1.0.0",
        "opencv-python>=4.5.0",
        "scikit-image>=0.19.0",
        "openai-whisper>=20230314",
        "librosa>=0.9.0",
        "tqdm>=4.64.0",
        "Pillow>=9.0.0",
        "requests>=2.25.0",
        "numba>=0.56.0",
        "psutil>=5.8.0",
        "ffmpeg-python>=0.2.0",
        "boto3>=1.26.0"
    ]
    
    success = True
    for dep in deps:
        if not run_pip_command([dep], f"dependency {dep}", required=False):
            success = False
    
    return success

def install_aeneas():
    """Install aeneas with multiple fallback strategies"""
    logger.info("4️⃣ Installing aeneas (with fallback strategies)...")
    
    strategies = [
        (["--no-build-isolation", "aeneas>=1.7.3"], "aeneas (no-build-isolation)"),
        (["aeneas>=1.7.3"], "aeneas (standard)"),
        (["git+https://github.com/readbeyond/aeneas.git"], "aeneas (from git)"),
        (["aeneas==1.7.3.0"], "aeneas (specific version)"),
    ]
    
    for args, description in strategies:
        logger.info(f"   Trying: {description}")
        if run_pip_command(args, description, required=False):
            logger.info("✅ aeneas installed successfully")
            return True
        logger.warning(f"   Failed: {description}")
    
    logger.error("❌ All aeneas installation strategies failed")
    logger.warning("   The system will work without aeneas but audio sync may be limited")
    return False

def test_imports():
    """Test if critical packages can be imported"""
    logger.info("5️⃣ Testing package imports...")
    
    packages = [
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("torch", "PyTorch"),
        ("cv2", "OpenCV"),
        ("whisper", "OpenAI Whisper"),
        ("runpod", "RunPod SDK"),
        ("requests", "Requests"),
    ]
    
    optional_packages = [
        ("aeneas", "Aeneas"),
    ]
    
    all_critical_good = True
    
    # Test critical packages
    for import_name, package_name in packages:
        try:
            __import__(import_name)
            logger.info(f"✅ {package_name}")
        except ImportError as e:
            logger.error(f"❌ {package_name} - CRITICAL: {e}")
            all_critical_good = False
    
    # Test optional packages
    for import_name, package_name in optional_packages:
        try:
            __import__(import_name)
            logger.info(f"✅ {package_name} (optional)")
        except ImportError as e:
            logger.warning(f"⚠️ {package_name} (optional) - {e}")
    
    return all_critical_good

def test_vantage_modules():
    """Test if VantageVeoConverter modules can be imported"""
    logger.info("6️⃣ Testing VantageVeoConverter modules...")
    
    modules = [
        ("src.comfy_rife", "ComfyRIFE"),
        ("src.audio_sync", "Audio Sync"),
        ("src.binary_utils", "Binary Utils"),
        ("src.physical_retime", "Physical Retime"),
    ]
    
    success = True
    for module_name, display_name in modules:
        try:
            __import__(module_name)
            logger.info(f"✅ {display_name}")
        except ImportError as e:
            logger.error(f"❌ {display_name} - {e}")
            success = False
    
    return success

def main():
    """Main installation function"""
    logger.info("🚀 VantageVeoConverter RunPod Dependency Installer")
    logger.info("=" * 60)
    
    # Check system deps
    missing_sys = check_system_dependencies()
    if missing_sys:
        logger.warning(f"Missing system dependencies: {missing_sys}")
        logger.warning("Some features may not work properly")
    
    # Install steps
    steps = [
        ("Core Dependencies", install_core_dependencies),
        ("PyTorch", install_pytorch), 
        ("Main Dependencies", install_main_dependencies),
        ("Aeneas (optional)", install_aeneas),
    ]
    
    failed_steps = []
    for step_name, step_func in steps:
        logger.info(f"\n{'='*20} {step_name} {'='*20}")
        if not step_func():
            failed_steps.append(step_name)
            logger.error(f"❌ {step_name} failed")
        else:
            logger.info(f"✅ {step_name} completed")
    
    # Test imports
    logger.info(f"\n{'='*20} Testing Installation {'='*20}")
    critical_imports_ok = test_imports()
    vantage_modules_ok = test_vantage_modules()
    
    # Summary
    logger.info(f"\n{'='*20} Installation Summary {'='*20}")
    if failed_steps:
        logger.warning(f"⚠️ Failed steps: {', '.join(failed_steps)}")
    
    if critical_imports_ok and vantage_modules_ok:
        logger.info("🎉 Installation completed successfully!")
        logger.info("✅ Ready for RunPod deployment")
        return True
    elif critical_imports_ok:
        logger.warning("⚠️ Installation mostly successful")
        logger.warning("Some optional features may not work")
        return True
    else:
        logger.error("❌ Critical installation failures detected")
        logger.error("Manual intervention required")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)