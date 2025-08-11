#!/usr/bin/env python3
"""
Automatic dependency installer for VantageVeoConverter
Handles problematic packages like aeneas correctly
"""
import subprocess
import sys
import os

def run_pip_command(args, description):
    """Run pip command with error handling"""
    print(f"📦 {description}...")
    
    cmd = [sys.executable, "-m", "pip", "install"] + args
    print(f"   Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ {description} - SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED")
        print(f"   Error: {e.stderr}")
        return False

def install_system_deps():
    """Install system dependencies for aeneas"""
    print("📦 Installing system dependencies for aeneas...")
    
    # Try apt-get (Ubuntu/Debian)
    try:
        cmd = ["apt-get", "update"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            cmd = ["apt-get", "install", "-y", "espeak", "libespeak-dev", "libasound2-dev", "build-essential"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print("✅ System dependencies installed via apt-get")
                return True
            else:
                print("⚠️ apt-get install failed, trying without sudo...")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        print("⚠️ apt-get not available or failed")
    
    # Try without sudo (Docker containers)
    try:
        cmd = ["apt", "update"]
        subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=True)
        cmd = ["apt", "install", "-y", "espeak", "libespeak-dev", "libasound2-dev", "build-essential"] 
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=True)
        print("✅ System dependencies installed via apt")
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        print("⚠️ apt not available")
    
    print("❌ Could not install system dependencies automatically")
    print("   Run manually: apt-get install espeak libespeak-dev libasound2-dev build-essential")
    return False

def install_dependencies():
    """Install all dependencies in correct order"""
    print("🚀 VantageVeoConverter Dependency Installer\n")
    
    # Install system dependencies first
    install_system_deps()
    print()
    
    # Step 1: Install modern numpy first
    print("1️⃣ Installing core dependencies...")
    if not run_pip_command([
        "numpy>=1.24,<2.3",
        "scipy>=1.9.0"
    ], "Core libraries (numpy, scipy)"):
        return False
    
    # Step 2: Install main requirements (without aeneas)
    print("\n2️⃣ Installing main dependencies...")
    if not run_pip_command(["-r", "requirements.txt"], "Main dependencies"):
        print("⚠️ Some dependencies failed, continuing...")
    
    # Step 3: Install aeneas with multiple strategies
    print("\n3️⃣ Installing aeneas...")
    
    aeneas_installed = False
    
    # Strategy 1: Try newest aeneas version (might be fixed)
    print("   Trying latest aeneas...")
    if run_pip_command(["aeneas"], "aeneas (latest)"):
        aeneas_installed = True
    
    # Strategy 2: Install from git (development version)
    if not aeneas_installed:
        print("   Trying aeneas from git...")
        if run_pip_command(["git+https://github.com/readbeyond/aeneas.git"], "aeneas (git)"):
            aeneas_installed = True
    
    # Strategy 3: Try conda-forge version via pip
    if not aeneas_installed:
        print("   Trying to install via alternative method...")
        # Install build dependencies first
        run_pip_command(["setuptools<60", "wheel", "cython"], "build tools")
        if run_pip_command(["aeneas>=1.7.3"], "aeneas (with build tools)"):
            aeneas_installed = True
    
    # Strategy 4: Try conda if available
    if not aeneas_installed:
        print("   Trying conda if available...")
        try:
            result = subprocess.run(["conda", "install", "-c", "conda-forge", "-y", "aeneas"], 
                                  capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print("✅ aeneas installed via conda")
                aeneas_installed = True
            else:
                print("   conda install failed")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            print("   conda not available")
    
    if not aeneas_installed:
        print("❌ Could not install aeneas. Manual install:")
        print("   pip install numpy==1.21.1 && pip install aeneas")
        return False
    
    print("\n✅ All dependencies installed successfully!")
    return True

def check_installation():
    """Check if key packages are importable"""
    print("\n🔍 Checking installation...")
    
    packages = [
        ("numpy", "numpy"),
        ("scipy", "scipy"), 
        ("torch", "torch"),
        ("cv2", "opencv-python"),
        ("whisper", "openai-whisper"),
        ("gradio", "gradio"),
        ("aeneas", "aeneas")
    ]
    
    all_good = True
    for import_name, package_name in packages:
        try:
            __import__(import_name)
            print(f"✅ {package_name}")
        except ImportError as e:
            print(f"❌ {package_name} - {e}")
            all_good = False
    
    if all_good:
        print("\n🎉 All packages working!")
    else:
        print("\n⚠️ Some packages have issues")
    
    return all_good

def main():
    """Main installer function"""
    if not install_dependencies():
        print("\n❌ Installation failed!")
        sys.exit(1)
    
    if not check_installation():
        print("\n⚠️ Installation completed but some packages have issues")
        sys.exit(1)
    
    print("\n🎯 Installation complete! Run 'python app_rife_compact.py' to start.")

if __name__ == "__main__":
    main()