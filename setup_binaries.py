#!/usr/bin/env python3
"""
Setup script to copy binaries for deployment
"""
import os
import shutil
import sys
import subprocess
import urllib.request
import tarfile

def download_ffmpeg_static():
    """Download FFmpeg static binary for Linux"""
    print("📥 Downloading FFmpeg static binary...")
    
    url = "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
    filename = "ffmpeg-static.tar.xz"
    
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"✅ Downloaded {filename}")
        
        # Extract
        with tarfile.open(filename, 'r:xz') as tar:
            tar.extractall()
        
        # Find extracted directory
        for item in os.listdir('.'):
            if item.startswith('ffmpeg-') and os.path.isdir(item):
                ffmpeg_dir = item
                break
        else:
            raise Exception("Could not find extracted FFmpeg directory")
        
        # Copy binaries
        bin_dir = "bin"
        os.makedirs(bin_dir, exist_ok=True)
        
        for binary in ['ffmpeg', 'ffprobe']:
            src = os.path.join(ffmpeg_dir, binary)
            dst = os.path.join(bin_dir, binary)
            if os.path.exists(src):
                shutil.copy2(src, dst)
                os.chmod(dst, 0o755)
                print(f"✅ Copied {binary}")
        
        # Cleanup
        os.remove(filename)
        shutil.rmtree(ffmpeg_dir)
        
        return True
        
    except Exception as e:
        print(f"❌ FFmpeg download failed: {e}")
        return False

def setup_system_binaries():
    """Try to use system binaries"""
    print("🔍 Looking for system binaries...")
    
    bin_dir = "bin"
    os.makedirs(bin_dir, exist_ok=True)
    
    binaries = ['ffmpeg', 'ffprobe', 'mp4fpsmod']
    found = []
    
    for binary in binaries:
        system_path = shutil.which(binary)
        if system_path:
            dst = os.path.join(bin_dir, binary)
            try:
                shutil.copy2(system_path, dst)
                os.chmod(dst, 0o755)
                print(f"✅ Copied {binary} from {system_path}")
                found.append(binary)
            except Exception as e:
                print(f"⚠️ Could not copy {binary}: {e}")
    
    return found

def main():
    """Main setup function"""
    print("🚀 Setting up VantageVeoConverter binaries...")
    
    # Try to copy from system first
    found = setup_system_binaries()
    
    # If FFmpeg not found in system, try to download
    if 'ffmpeg' not in found or 'ffprobe' not in found:
        if download_ffmpeg_static():
            print("✅ FFmpeg downloaded successfully")
        else:
            print("❌ Could not get FFmpeg - you'll need to install it manually")
    
    # Check what we have
    bin_dir = "bin"
    print(f"\n📁 Final binary status in {bin_dir}/:")
    
    for binary in ['ffmpeg', 'ffprobe', 'mp4fpsmod']:
        path = os.path.join(bin_dir, binary)
        if os.path.exists(path):
            size = os.path.getsize(path) // 1024 // 1024
            print(f"✅ {binary} - {size}MB")
        else:
            print(f"❌ {binary} - missing")
    
    print(f"\n🎯 Setup complete! Run 'python app_rife_compact.py' to start.")

if __name__ == "__main__":
    main()