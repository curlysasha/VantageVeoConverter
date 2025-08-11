#!/usr/bin/env python3
"""
Test script to verify binary detection
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.getcwd())

from src.binary_utils import get_ffmpeg, get_ffprobe, get_mp4fpsmod

def test_binary(name, get_func):
    """Test a binary function"""
    print(f"🔍 Testing {name}...")
    
    try:
        path = get_func()
        if path:
            print(f"✅ {name}: {path}")
            
            # Test execution
            if os.path.exists(path):
                size = os.path.getsize(path) // 1024 // 1024
                print(f"   Size: {size}MB")
                
                # Try to run with --version
                import subprocess
                try:
                    if name == 'mp4fpsmod':
                        # mp4fpsmod doesn't have --version, just run it
                        result = subprocess.run([path], capture_output=True, text=True, timeout=5)
                        if 'mp4fpsmod' in result.stderr:
                            print(f"   Status: Working ✅")
                        else:
                            print(f"   Status: Unknown response")
                    else:
                        result = subprocess.run([path, '-version'], capture_output=True, text=True, timeout=5)
                        if result.returncode == 0:
                            first_line = result.stdout.split('\n')[0]
                            print(f"   Version: {first_line}")
                            print(f"   Status: Working ✅")
                        else:
                            print(f"   Status: Error {result.returncode}")
                except subprocess.TimeoutExpired:
                    print(f"   Status: Timeout (but binary exists)")
                except Exception as e:
                    print(f"   Status: Test failed - {e}")
            else:
                print(f"   Status: Path returned but file doesn't exist ❌")
        else:
            print(f"❌ {name}: Not found")
            
    except Exception as e:
        print(f"❌ {name}: Error - {e}")
    
    print()

def main():
    """Main test function"""
    print("🧪 VantageVeoConverter Binary Test\n")
    
    print(f"Working directory: {os.getcwd()}")
    print(f"Script location: {os.path.abspath(__file__)}")
    print()
    
    # Test each binary
    test_binary('ffmpeg', get_ffmpeg)
    test_binary('ffprobe', get_ffprobe) 
    test_binary('mp4fpsmod', get_mp4fpsmod)
    
    print("🎯 Test complete!")

if __name__ == "__main__":
    main()