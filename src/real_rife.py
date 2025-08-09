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
            
            # Try to load pretrained weights
            self._load_pretrained_weights()
            
            logging.info("✅ RIFE model initialized")
            
        except Exception as e:
            logging.error(f"Model loading failed: {e}")
            raise Exception("❌ RIFE model loading failed!")
    
    def _load_pretrained_weights(self):
        """Load pretrained RIFE weights."""
        try:
            # Try to find weights in project weights directory first, then RIFE directory
            project_root = os.path.dirname(os.path.dirname(__file__))  # Go up from src/ to project root
            possible_weights = [
                # Project weights directory (preferred)
                os.path.join(project_root, 'weights', 'flownet.pkl'),
                os.path.join(project_root, 'weights', 'flownet.pth'),
                os.path.join(project_root, 'weights', 'rife_model.pkl'),
                os.path.join(project_root, 'weights', 'rife_model.pth'),
                # RIFE directory (fallback)
                os.path.join(self.rife_dir, 'train_log', 'flownet.pkl'),
                os.path.join(self.rife_dir, 'checkpoints', 'flownet.pkl'),
                os.path.join(self.rife_dir, 'flownet.pkl'),
            ]
            
            weights_loaded = False
            for weight_path in possible_weights:
                if os.path.exists(weight_path):
                    logging.info(f"🔄 Loading RIFE weights from: {weight_path}")
                    try:
                        if weight_path.endswith('.pkl'):
                            import pickle
                            # Try different pickle loading approaches
                            try:
                                with open(weight_path, 'rb') as f:
                                    weights = pickle.load(f)
                            except Exception as e1:
                                logging.warning(f"   Standard pickle load failed: {e1}")
                                # Try with torch.load which can handle torch tensors better
                                try:
                                    weights = torch.load(weight_path, map_location=self.device, pickle_module=pickle)
                                except Exception as e2:
                                    logging.warning(f"   Torch pickle load failed: {e2}")
                                    # Try with encoding fix
                                    weights = torch.load(weight_path, map_location=self.device, encoding='latin1')
                        else:
                            weights = torch.load(weight_path, map_location=self.device)
                        
                        logging.info(f"   Weights type: {type(weights)}")
                        
                        # Handle different weight formats
                        if isinstance(weights, dict):
                            # Standard dict format
                            state_dict = weights
                        elif isinstance(weights, int):
                            # This is likely an epoch number, skip
                            logging.warning(f"   File contains epoch number ({weights}), not weights - skipping")
                            continue
                        elif hasattr(weights, 'state_dict'):
                            # Model checkpoint format
                            state_dict = weights.state_dict()
                        else:
                            logging.warning(f"   Unknown weights format: {type(weights)}")
                            continue
                        
                        logging.info(f"   State dict keys: {len(state_dict)} items")
                        
                        # Try different loading approaches
                        load_success = False
                        
                        if hasattr(self.model, 'load_state_dict'):
                            try:
                                # Try to load compatible weights only
                                result = self.model.load_state_dict(state_dict, strict=False)
                                if result.missing_keys or result.unexpected_keys:
                                    logging.info(f"   Partial load: {len(result.missing_keys)} missing, {len(result.unexpected_keys)} unexpected")
                                    if len(result.missing_keys) < len(state_dict) // 2:  # If we loaded at least 50%
                                        load_success = True
                                        logging.info("   ✅ Partial weights loaded successfully (>50% compatible)")
                                    else:
                                        logging.warning("   Too many missing keys for useful partial load")
                                else:
                                    load_success = True
                                    logging.info("   ✅ Perfect weight match")
                            except Exception as e:
                                logging.warning(f"   Model load_state_dict failed: {e}")
                        
                        if not load_success and hasattr(self.model, 'flownet'):
                            try:
                                result = self.model.flownet.load_state_dict(state_dict, strict=False)
                                if result.missing_keys or result.unexpected_keys:
                                    logging.info(f"   Flownet partial load: {len(result.missing_keys)} missing, {len(result.unexpected_keys)} unexpected")
                                    if len(result.missing_keys) < len(state_dict) // 2:
                                        load_success = True
                                        logging.info("   ✅ Flownet partial weights loaded successfully")
                                    else:
                                        logging.warning("   Too many missing flownet keys")
                                else:
                                    load_success = True
                                    logging.info("   ✅ Perfect flownet weight match")
                            except Exception as e:
                                logging.warning(f"   Flownet load_state_dict failed: {e}")
                        
                        if load_success:
                            logging.info("✅ RIFE pretrained weights loaded successfully!")
                            weights_loaded = True
                            break
                        else:
                            logging.warning("⚠️ Failed to apply weights to model")
                            continue
                        
                    except Exception as e:
                        logging.warning(f"⚠️ Failed to load weights from {weight_path}: {e}")
                        continue
            
            if not weights_loaded:
                logging.warning("⚠️ No pretrained weights found - using random initialization")
                logging.warning("   This will cause poor interpolation quality!")
                logging.info("   To fix: Download PROPER RIFE weights")
                logging.info("   Try these files from RIFE GitHub:")
                logging.info("   - train_log/flownet.pkl (not the root flownet.pkl)")  
                logging.info("   - Or contextnet.pkl + unet.pkl")
                logging.info(f"   Place in: {os.path.join(project_root, 'weights', 'flownet.pkl')}")
                
        except Exception as e:
            logging.warning(f"⚠️ Weight loading error: {e}")
            logging.warning("   Using random weights - expect poor quality")
    
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
        """Interpolate single frame at specific timestep using REAL RIFE or fallback."""
        if not self.available or self.model is None:
            logging.warning("⚠️ REAL RIFE not available, using OpenCV interpolation fallback")
            return self._opencv_interpolate(frame1, frame2, timestep)
        
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
            logging.warning(f"REAL RIFE timestep interpolation failed: {e}, falling back to OpenCV")
            return self._opencv_interpolate(frame1, frame2, timestep)
    
    def _opencv_interpolate(self, frame1, frame2, timestep):
        """Fallback interpolation using OpenCV optical flow."""
        try:
            # Simple weighted blend as basic interpolation
            weight1 = 1.0 - timestep
            weight2 = timestep
            
            # Ensure same data types
            frame1_f = frame1.astype(np.float32)
            frame2_f = frame2.astype(np.float32)
            
            # Linear blend
            result = (frame1_f * weight1 + frame2_f * weight2)
            result = np.clip(result, 0, 255).astype(np.uint8)
            
            return result
            
        except Exception as e:
            logging.error(f"OpenCV interpolation failed: {e}")
            # Last resort - return frame1
            return frame1.copy()
    
    def _frame_to_tensor(self, frame):
        """Convert BGR frame to RGB tensor with proper size alignment."""
        # BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Store original dimensions for later cropping
        self.orig_h, self.orig_w = frame_rgb.shape[:2]
        
        # RIFE requires dimensions divisible by 64
        h, w = frame_rgb.shape[:2]
        pad_h = ((h + 63) // 64) * 64 - h
        pad_w = ((w + 63) // 64) * 64 - w
        
        # Store padding for later removal
        self.pad_h, self.pad_w = pad_h, pad_w
        
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
        """Convert RGB tensor to BGR frame and remove padding."""
        # Move to CPU and convert back to frame
        tensor = tensor.squeeze(0).cpu().permute(1, 2, 0)  # BCHW -> CHW -> HWC
        frame_rgb = (tensor.clamp(0, 1) * 255).byte().numpy()
        
        # Remove padding to restore original size
        if hasattr(self, 'pad_h') and hasattr(self, 'pad_w'):
            if self.pad_h > 0 or self.pad_w > 0:
                # Crop back to original size
                h_end = self.orig_h
                w_end = self.orig_w
                frame_rgb = frame_rgb[:h_end, :w_end]
        
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        return frame_bgr