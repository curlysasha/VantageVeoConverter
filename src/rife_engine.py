"""
RIFE AI Engine - Clean wrapper for Real RIFE
"""
import logging
import torch
from .comfy_rife import ComfyRIFE as RealRIFEModel

class RealRIFE:
    """Clean wrapper for Real RIFE implementation."""
    
    def __init__(self, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.available = False
        self.real_rife_model = None
        self._setup_rife()
    
    def _setup_rife(self):
        """Setup REAL RIFE."""
        try:
            logging.info("🤖 Setting up REAL RIFE engine...")
            
            # Initialize REAL RIFE
            self.real_rife_model = RealRIFEModel(self.device)
            
            if self.real_rife_model.available:
                self.available = True
                logging.info("✅ REAL RIFE engine ready!")
                
                # GPU info
                if torch.cuda.is_available():
                    logging.info(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
                    torch.cuda.empty_cache()
            else:
                raise Exception("❌ REAL RIFE failed!")
                
        except Exception as e:
            logging.error(f"❌ REAL RIFE engine setup failed: {e}")
            raise Exception(f"❌ REAL RIFE engine setup failed: {e}")
    
    def interpolate_frames(self, frame1, frame2, num_intermediate=1):
        """REAL RIFE interpolation."""
        if not self.available or not self.real_rife_model:
            raise Exception("❌ REAL RIFE not available!")
        
        return self.real_rife_model.interpolate_frames(frame1, frame2, num_intermediate)