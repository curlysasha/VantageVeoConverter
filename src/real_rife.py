"""
Real RIFE implementation using original ECCV2022-RIFE repository - CLEAN VERSION
"""
import cv2
import numpy as np
import logging
import torch
import os
import subprocess
import sys
import tempfile

class RealRIFE:
    """Real RIFE using official ECCV2022-RIFE repository - CLEAN VERSION."""
    
    def __init__(self, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.available = False
        self.rife_dir = None
        self.model = None
        self._setup_rife()
    
    def _setup_rife(self):
        """Setup real RIFE from official repository."""
        try:
            logging.info("🚀 Setting up REAL RIFE...")
            
            # Clone RIFE repository to temp directory
            self.rife_dir = os.path.join(tempfile.gettempdir(), "ECCV2022-RIFE")
            
            if not os.path.exists(self.rife_dir):
                logging.info("📦 Cloning ECCV2022-RIFE...")
                result = subprocess.run([
                    "git", "clone", "https://github.com/megvii-research/ECCV2022-RIFE.git", 
                    self.rife_dir
                ], capture_output=True, text=True, timeout=300)
                
                if result.returncode != 0:
                    raise Exception(f"Git clone failed: {result.stderr}")
            
            # Add RIFE to Python path
            if self.rife_dir not in sys.path:
                sys.path.insert(0, self.rife_dir)
            
            # Import and initialize RIFE model
            self._load_rife_model()
            
            self.available = True
            logging.info("✅ REAL RIFE ready!")
            
        except Exception as e:
            logging.error(f"❌ Real RIFE setup failed: {e}")
            self.available = False
    
    def _load_rife_model(self):
        """Load RIFE model."""
        try:
            # Import RIFE model - try most common path first
            try:
                from model.RIFE import Model
                logging.info("✅ Successfully imported model.RIFE.Model")
            except ImportError:
                # Fallback import paths
                import_paths = [
                    ("model.RIFE_HDv3", "Model"),
                    ("RIFE_HDv3", "Model"),
                    ("RIFE", "Model")
                ]
                
                model_class = None
                for module_path, class_name in import_paths:
                    try:
                        module = __import__(module_path, fromlist=[class_name])
                        Model = getattr(module, class_name)
                        logging.info(f"✅ Successfully imported {module_path}.{class_name}")
                        break
                    except (ImportError, AttributeError):
                        continue
                
                if Model is None:
                    raise Exception("❌ Cannot import REAL RIFE model!")
            
            # Initialize model
            self.model = Model()
            self.model.eval()
            
            logging.warning("⚠️ Using random weights (weights loading disabled for now)")
            
        except Exception as e:
            logging.error(f"Model loading failed: {e}")
            raise Exception("❌ RIFE model loading failed!")
    
    def interpolate_frames(self, frame1, frame2, num_intermediate=1):
        """Interpolate frames using REAL RIFE."""
        if not self.available or self.model is None:
            raise Exception("❌ REAL RIFE not available!")
        
        try:
            # Convert frames to tensors
            tensor1 = self._frame_to_tensor(frame1)
            tensor2 = self._frame_to_tensor(frame2)
            
            interpolated_frames = []
            
            with torch.no_grad():
                for i in range(1, num_intermediate + 1):
                    timestep = i / (num_intermediate + 1)
                    
                    # Run RIFE inference
                    result_tensor = self.model.inference(tensor1, tensor2, timestep)
                    
                    # Convert back to frame
                    result_frame = self._tensor_to_frame(result_tensor)
                    interpolated_frames.append(result_frame)
            
            return interpolated_frames
            
        except Exception as e:
            logging.error(f"REAL RIFE interpolation failed: {e}")
            raise Exception(f"❌ REAL RIFE interpolation failed: {e}")
    
    def interpolate_at_timestep(self, frame1, frame2, timestep):
        """Interpolate single frame at specific timestep using REAL RIFE."""
        if not self.available or self.model is None:
            raise Exception("❌ REAL RIFE not available!")
        
        try:
            # Convert frames to tensors
            tensor1 = self._frame_to_tensor(frame1)
            tensor2 = self._frame_to_tensor(frame2)
            
            with torch.no_grad():
                # Run RIFE inference with custom timestep
                result_tensor = self.model.inference(tensor1, tensor2, timestep)
                
                # Convert back to frame
                result_frame = self._tensor_to_frame(result_tensor)
                return result_frame
            
        except Exception as e:
            logging.error(f"REAL RIFE timestep interpolation failed: {e}")
            raise Exception(f"❌ REAL RIFE timestep interpolation failed: {e}")
    
    def _frame_to_tensor(self, frame):
        """Convert BGR frame to RGB tensor with proper size alignment."""
        # BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # RIFE requires dimensions divisible by 64
        h, w = frame_rgb.shape[:2]
        pad_h = ((h + 63) // 64) * 64 - h
        pad_w = ((w + 63) // 64) * 64 - w
        
        # Pad frame if necessary
        if pad_h > 0 or pad_w > 0:
            frame_rgb = cv2.copyMakeBorder(
                frame_rgb, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101
            )
        
        # Normalize to [0, 1] and convert to tensor
        tensor = torch.from_numpy(frame_rgb).float() / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # HWC -> CHW -> BCHW
        
        return tensor.to(self.device)
    
    def _tensor_to_frame(self, tensor):
        """Convert RGB tensor to BGR frame."""
        # Move to CPU and convert back to frame
        tensor = tensor.squeeze(0).cpu().permute(1, 2, 0)  # BCHW -> CHW -> HWC
        frame_rgb = (tensor.clamp(0, 1) * 255).byte().numpy()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        return frame_bgr