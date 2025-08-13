#!/usr/bin/env python3
"""
VantageVeoConverter RunPod Local Testing Script
Тестирует handler локально без RunPod инфраструктуры
"""

import sys
import json
import time
import tempfile
import os
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_handler_import():
    """Test that handler can be imported"""
    print("🧪 Testing handler import...")
    try:
        from runpod_handler import handler, initialize_models
        print("✅ Handler imported successfully")
        return True
    except Exception as e:
        print(f"❌ Handler import failed: {e}")
        return False

def test_dependencies():
    """Test that all dependencies are available"""
    print("\n🧪 Testing dependencies...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch: {e}")
        return False
    
    try:
        import whisper
        print(f"✅ Whisper available")
    except ImportError as e:
        print(f"❌ Whisper: {e}")
        return False
    
    try:
        import cv2
        print(f"✅ OpenCV: {cv2.__version__}")
    except ImportError as e:
        print(f"❌ OpenCV: {e}")
        return False
    
    try:
        import runpod
        print(f"✅ RunPod SDK available")
    except ImportError as e:
        print(f"❌ RunPod SDK: {e}")
        return False
    
    # Test src modules
    try:
        from src.comfy_rife import ComfyRIFE
        from src.audio_sync import synchronize_audio_video_workflow
        from src.binary_utils import check_all_binaries
        print("✅ VantageVeoConverter modules available")
    except ImportError as e:
        print(f"❌ VantageVeoConverter modules: {e}")
        return False
    
    return True

def test_models_initialization():
    """Test model initialization"""
    print("\n🧪 Testing model initialization...")
    
    try:
        from runpod_handler import initialize_models, WHISPER_MODEL, RIFE_MODEL
        
        # This should load models
        initialize_models()
        
        print(f"✅ Whisper model loaded: {WHISPER_MODEL is not None}")
        print(f"✅ RIFE model loaded: {RIFE_MODEL is not None}")
        
        return True
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        return False

def test_handler_with_mock_data():
    """Test handler with mock job data"""
    print("\n🧪 Testing handler with mock data...")
    
    # Mock job data
    mock_job = {
        "id": "test_job_12345",
        "input": {
            "video_url": "https://sample-videos.com/zip/10/mp4/SampleVideo_1280x720_1mb.mp4",
            "audio_url": "https://www2.cs.uic.edu/~i101/SoundFiles/BabyElephantWalk60.wav",
            "use_rife": False,  # Disable RIFE for faster testing
            "diagnostic_mode": False
        }
    }
    
    try:
        from runpod_handler import handler
        
        print("🚀 Starting handler test...")
        start_time = time.time()
        
        result = handler(mock_job)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"⏱️  Processing time: {processing_time:.2f}s")
        print(f"📊 Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
        
        if isinstance(result, dict) and result.get("success"):
            print("✅ Handler test passed")
            print(f"📹 Video URL type: {'base64' if result.get('synchronized_video_url', '').startswith('data:') else 'URL'}")
            return True
        else:
            print(f"❌ Handler test failed: {result}")
            return False
            
    except Exception as e:
        print(f"❌ Handler test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_binary_dependencies():
    """Test binary dependencies (ffmpeg, mp4fpsmod)"""
    print("\n🧪 Testing binary dependencies...")
    
    try:
        from src.binary_utils import check_all_binaries
        missing = check_all_binaries()
        
        if not missing:
            print("✅ All binary dependencies available")
            return True
        else:
            print(f"❌ Missing binary dependencies: {missing}")
            return False
            
    except Exception as e:
        print(f"❌ Binary dependency check failed: {e}")
        return False

def create_test_files():
    """Create small test files for local testing"""
    print("\n🧪 Creating test files...")
    
    try:
        # Create a small test video using ffmpeg
        import subprocess
        
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, "test_video.mp4")
        audio_path = os.path.join(temp_dir, "test_audio.wav")
        
        # Generate 5-second test video
        cmd_video = [
            "ffmpeg", "-f", "lavfi", "-i", "testsrc=duration=5:size=320x240:rate=30",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-y", video_path
        ]
        
        # Generate 5-second test audio  
        cmd_audio = [
            "ffmpeg", "-f", "lavfi", "-i", "sine=frequency=1000:duration=5",
            "-c:a", "pcm_s16le", "-y", audio_path
        ]
        
        print("🎬 Generating test video...")
        result = subprocess.run(cmd_video, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Video generation failed: {result.stderr}")
            return None, None
            
        print("🎵 Generating test audio...")
        result = subprocess.run(cmd_audio, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Audio generation failed: {result.stderr}")
            return None, None
        
        print(f"✅ Test files created in {temp_dir}")
        return video_path, audio_path
        
    except Exception as e:
        print(f"❌ Test file creation failed: {e}")
        return None, None

def run_all_tests():
    """Run all tests"""
    print("🚀 VantageVeoConverter RunPod Local Testing")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_handler_import),
        ("Dependencies Test", test_dependencies),
        ("Binary Dependencies Test", test_binary_dependencies),
        ("Model Initialization Test", test_models_initialization),
        ("Handler Mock Test", test_handler_with_mock_data),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name}...")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"💥 {test_name} CRASHED: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Ready for RunPod deployment.")
        return True
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)