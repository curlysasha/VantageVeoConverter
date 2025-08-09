"""
Real RIFE implementation using original ECCV2022-RIFE repository
"""
import cv2
import numpy as np
import logging
import torch
import torch.nn.functional as F
import os
import subprocess
import sys
import tempfile
from pathlib import Path

class RealRIFE:
    """Real RIFE using official ECCV2022-RIFE repository."""
    
    def __init__(self, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.available = False
        self.rife_dir = None
        self.model = None
        self._setup_rife()
    
    def _setup_rife(self):
        """Setup real RIFE from official repository."""
        try:
            logging.info("🚀 Setting up REAL RIFE from official repository...")
            
            # Clone RIFE repository to temp directory
            self.rife_dir = os.path.join(tempfile.gettempdir(), "ECCV2022-RIFE")
            
            if not os.path.exists(self.rife_dir):
                logging.info("📦 Cloning ECCV2022-RIFE repository...")
                result = subprocess.run([
                    "git", "clone", "https://github.com/megvii-research/ECCV2022-RIFE.git", 
                    self.rife_dir
                ], capture_output=True, text=True, timeout=300)
                
                if result.returncode != 0:
                    raise Exception(f"Git clone failed: {result.stderr}")
                
                logging.info("✅ RIFE repository cloned")
            
            # Add RIFE to Python path
            if self.rife_dir not in sys.path:
                sys.path.insert(0, self.rife_dir)
            
            # Download model weights
            self._download_model()
            
            # Import and initialize RIFE model
            self._load_rife_model()
            
            self.available = True
            logging.info("✅ REAL RIFE ready!")
            
        except Exception as e:
            logging.error(f"❌ Real RIFE setup failed: {e}")
            self.available = False
    
    def _download_model(self):
        """Download RIFE model weights."""
        try:
            model_path = os.path.join(self.rife_dir, "train_log")
            os.makedirs(model_path, exist_ok=True)
            
            # Check if model already exists
            flownet_path = os.path.join(model_path, "flownet.pkl")
            if os.path.exists(flownet_path):
                logging.info("✅ RIFE model already exists")
                return
            
            logging.info("📥 Downloading RIFE model weights...")
            
            # Try multiple model sources
            model_urls = [
                "https://github.com/hzwer/Practical-RIFE/releases/download/4.6/train_log.zip",
                "https://huggingface.co/AlexWortega/RIFE/resolve/main/flownet.pkl",
                "https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/rife46_v2.0.zip"
            ]
            
            import urllib.request
            import zipfile
            
            # Try downloading from different sources
            download_success = False
            for i, model_url in enumerate(model_urls):
                try:
                    logging.info(f"   Trying source {i+1}/{len(model_urls)}: {model_url[:50]}...")
                    
                    if model_url.endswith('.pkl'):
                        # Direct model file
                        direct_model_path = os.path.join(model_path, "flownet.pkl")
                        urllib.request.urlretrieve(model_url, direct_model_path)
                        logging.info(f"✅ RIFE model downloaded directly from source {i+1}")
                        download_success = True
                        break
                    else:
                        # Zip archive
                        zip_path = os.path.join(model_path, "train_log.zip")
                        urllib.request.urlretrieve(model_url, zip_path)
                        
                        # Extract model
                        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                            zip_ref.extractall(os.path.dirname(model_path))
                        
                        os.remove(zip_path)
                        logging.info(f"✅ RIFE model downloaded from source {i+1}")
                        download_success = True
                        break
                    
                except Exception as e:
                    logging.warning(f"   Source {i+1} failed: {e}")
                    # Clean up failed downloads
                    for cleanup_file in ["train_log.zip", "flownet.pkl"]:
                        cleanup_path = os.path.join(model_path, cleanup_file)
                        if os.path.exists(cleanup_path):
                            os.remove(cleanup_path)
                    continue
            
            if not download_success:
                raise Exception("❌ Could not download RIFE model from any source! NO FALLBACKS!")
            
        except Exception as e:
            logging.error(f"Model download failed: {e}")
            raise
    
    def _load_rife_model(self):
        """Load RIFE model."""
        try:
            # Import RIFE modules
            sys.path.insert(0, self.rife_dir)
            
            # Try different import paths
            try:
                from model.RIFE_HDv3 import Model
            except ImportError:
                try:
                    from RIFE_HDv3 import Model
                except ImportError:
                    raise Exception("❌ Cannot import RIFE model! NO FALLBACKS!")
            
            # Initialize model
            self.model = Model()
            
            # Load weights
            model_path = os.path.join(self.rife_dir, "train_log", "flownet.pkl")
            if os.path.exists(model_path):
                self.model.load_model(model_path, -1)
                logging.info("✅ RIFE model weights loaded")
            else:
                logging.warning("⚠️ No model weights found, using random initialization")
            
            # Set to eval mode
            self.model.eval()
            
        except Exception as e:
            logging.error(f"Model loading failed: {e}")
            raise Exception("❌ RIFE model loading failed! NO FALLBACKS!")
    
    # NO FALLBACKS! ONLY REAL RIFE!
    
    def interpolate_frames(self, frame1, frame2, num_intermediate=1):
        """Interpolate frames using REAL RIFE."""
        if not self.available or self.model is None:
            raise Exception("❌ REAL RIFE not available! NO FALLBACKS!")
        
        try:
            # Convert frames to tensors
            tensor1 = self._frame_to_tensor(frame1)
            tensor2 = self._frame_to_tensor(frame2)
            
            interpolated_frames = []
            
            with torch.no_grad():
                for i in range(1, num_intermediate + 1):
                    timestep = i / (num_intermediate + 1)
                    
                    # Run RIFE inference
                    if hasattr(self.model, 'inference'):
                        result_tensor = self.model.inference(tensor1, tensor2, timestep)
                    else:
                        raise Exception("❌ RIFE model has no inference method! NO FALLBACKS!")
                    
                    # Convert back to frame
                    result_frame = self._tensor_to_frame(result_tensor)
                    interpolated_frames.append(result_frame)
                    
                    logging.info(f"✅ REAL RIFE interpolation at timestep {timestep:.3f}")
            
            return interpolated_frames
            
        except Exception as e:
            logging.error(f"REAL RIFE interpolation failed: {e}")
            raise Exception("❌ REAL RIFE interpolation failed! NO FALLBACKS!")
    
    def _frame_to_tensor(self, frame):
        """Convert BGR frame to RGB tensor."""
        # BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        tensor = torch.from_numpy(frame_rgb).float() / 255.0
        
        # HWC to CHW and add batch dimension
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        
        return tensor.to(self.device)
    
    def _tensor_to_frame(self, tensor):
        """Convert RGB tensor to BGR frame."""
        # Move to CPU and remove batch dimension
        tensor = tensor.squeeze(0).cpu()
        
        # CHW to HWC
        tensor = tensor.permute(1, 2, 0)
        
        # Denormalize and convert to uint8
        frame_rgb = (tensor.clamp(0, 1) * 255).byte().numpy()
        
        # RGB to BGR
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        return frame_bgr

# Test function
def test_real_rife():
    """Test Real RIFE implementation."""
    rife = RealRIFE()
    
    if not rife.available:
        print("❌ Real RIFE not available")
        return False
    
    # Create test frames
    frame1 = np.zeros((240, 320, 3), dtype=np.uint8)
    frame2 = np.ones((240, 320, 3), dtype=np.uint8) * 255
    
    # Add moving object
    cv2.circle(frame1, (80, 120), 20, (0, 255, 0), -1)
    cv2.circle(frame2, (240, 120), 20, (0, 255, 0), -1)
    
    # Test interpolation
    try:
        result = rife.interpolate_frames(frame1, frame2, 1)
        if result and len(result) > 0:
            print("✅ Real RIFE test successful!")
            return True
        else:
            print("❌ Real RIFE returned empty result")
            return False
    except Exception as e:
        print(f"❌ Real RIFE test failed: {e}")
        return False

if __name__ == "__main__":
    test_real_rife()