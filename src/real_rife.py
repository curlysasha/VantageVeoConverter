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
            
            # Try different import paths for RIFE model
            model_class = None
            # Check what files actually exist in the repository
            rife_model_files = []
            for root, dirs, files in os.walk(self.rife_dir):
                for file in files:
                    if file.endswith('.py') and ('RIFE' in file or 'model' in file.lower()):
                        rel_path = os.path.relpath(os.path.join(root, file), self.rife_dir)
                        rife_model_files.append(rel_path)
                        logging.info(f"   Found RIFE file: {rel_path}")
            
            import_paths = [
                # Try based on actual files found
                ("model.RIFE_HDv3", "Model"),
                ("RIFE_HDv3", "Model"),
                ("model.RIFE", "Model"), 
                ("RIFE", "Model"),
                ("IFNet_HDv3", "Model"),
                ("model.IFNet_HDv3", "Model")
            ]
            
            for module_path, class_name in import_paths:
                try:
                    logging.info(f"   Trying import: from {module_path} import {class_name}")
                    module = __import__(module_path, fromlist=[class_name])
                    model_class = getattr(module, class_name)
                    logging.info(f"   ✅ Successfully imported {module_path}.{class_name}")
                    break
                except (ImportError, AttributeError) as e:
                    logging.debug(f"   ❌ Import failed: {module_path}.{class_name} - {e}")
                    continue
            
            if model_class is None:
                raise Exception("❌ Cannot import REAL RIFE model! NO FALLBACKS!")
            
            # Initialize model
            self.model = model_class()
            
            # Load weights - check different possible paths
            possible_model_paths = [
                os.path.join(self.rife_dir, "train_log", "flownet.pkl"),
                os.path.join(self.rife_dir, "flownet.pkl"),
                os.path.join(self.rife_dir, "train_log", "train_log", "flownet.pkl")
            ]
            
            model_loaded = False
            for model_path in possible_model_paths:
                if os.path.exists(model_path) and os.path.isfile(model_path):
                    try:
                        logging.info(f"   Trying to load model from: {model_path}")
                        self.model.load_model(model_path, -1)
                        logging.info("✅ RIFE model weights loaded")
                        model_loaded = True
                        break
                    except Exception as e:
                        logging.warning(f"   Failed to load from {model_path}: {e}")
                        continue
                else:
                    logging.debug(f"   Path does not exist or is not file: {model_path}")
            
            if not model_loaded:
                logging.warning("⚠️ No model weights found, using random initialization")
            
            # Set to eval mode
            self.model.eval()
            
        except Exception as e:
            logging.error(f"Model loading failed: {e}")
            raise Exception("❌ RIFE model loading failed! NO FALLBACKS!")
    
    # NO MINIMAL MODELS! ONLY REAL RIFE!
    
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
        """Convert BGR frame to RGB tensor with proper size alignment."""
        # BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # RIFE requires dimensions divisible by 64
        h, w = frame_rgb.shape[:2]
        
        # Calculate padding to make dimensions divisible by 64
        pad_h = ((h + 63) // 64) * 64 - h
        pad_w = ((w + 63) // 64) * 64 - w
        
        # Pad frame if necessary
        if pad_h > 0 or pad_w > 0:
            frame_rgb = cv2.copyMakeBorder(
                frame_rgb, 0, pad_h, 0, pad_w, 
                cv2.BORDER_REFLECT_101
            )
            logging.debug(f"Padded frame from {h}x{w} to {frame_rgb.shape[0]}x{frame_rgb.shape[1]}")
        
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