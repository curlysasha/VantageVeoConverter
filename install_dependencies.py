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

def install_dependencies():
    """Install all dependencies in correct order"""
    print("🚀 VantageVeoConverter Dependency Installer\n")
    
    # Step 1: Core dependencies first (older numpy for aeneas compatibility)
    print("1️⃣ Installing core dependencies...")
    if not run_pip_command([
        "numpy==1.21.1",  # Older numpy compatible with aeneas
        "scipy>=1.9.0"
    ], "Core libraries (numpy, scipy)"):
        return False
    
    # Step 2: Install main requirements (without aeneas)
    print("\n2️⃣ Installing main dependencies...")
    if not run_pip_command(["-r", "requirements.txt"], "Main dependencies"):
        print("⚠️ Some dependencies failed, continuing...")
    
    # Step 3: Install aeneas with special handling
    print("\n3️⃣ Installing aeneas with special handling...")
    
    # Try different approaches for aeneas
    aeneas_installed = False
    
    # Approach 1: --no-build-isolation
    print("   Trying --no-build-isolation...")
    if run_pip_command(["--no-build-isolation", "aeneas>=1.7.3"], "aeneas (no-build-isolation)"):
        aeneas_installed = True
    
    # Approach 2: --no-deps (if approach 1 failed)
    if not aeneas_installed:
        print("   Trying --no-deps...")
        if run_pip_command(["--no-deps", "aeneas>=1.7.3"], "aeneas (no-deps)"):
            aeneas_installed = True
    
    # Approach 3: Force reinstall older numpy then aeneas
    if not aeneas_installed:
        print("   Trying older numpy for compatibility...")
        run_pip_command(["--force-reinstall", "numpy==1.21.1"], "numpy (older version)")
        if run_pip_command(["aeneas>=1.7.3"], "aeneas (with older numpy)"):
            aeneas_installed = True
    
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