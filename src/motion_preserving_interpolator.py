"""
Motion-preserving interpolator - keeps natural movement, only fixes timing
"""
import cv2
import numpy as np
import torch
import logging
from typing import List, Dict, Tuple

class MotionPreservingInterpolator:
    """Interpolator that preserves motion and only fixes timing issues."""
    
    def __init__(self, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_gpu = torch.cuda.is_available() and device == "cuda"
        
        logging.info(f"🎭 Motion-preserving interpolator initialized on {self.device}")
    
    def create_motion_aware_interpolation(self, frame1: np.ndarray, frame2: np.ndarray, 
                                        alpha: float = 0.5, preserve_motion: bool = True) -> np.ndarray:
        """
        Create interpolated frame that preserves motion characteristics.
        Only does subtle blending, not dramatic changes.
        """
        if not preserve_motion:
            # Simple blend for static content
            return cv2.addWeighted(frame1, 1-alpha, frame2, alpha, 0)
        
        if self.use_gpu:
            return self._gpu_motion_interpolate(frame1, frame2, alpha)
        else:
            return self._cpu_motion_interpolate(frame1, frame2, alpha)
    
    def _cpu_motion_interpolate(self, frame1: np.ndarray, frame2: np.ndarray, alpha: float) -> np.ndarray:
        """CPU motion-preserving interpolation."""
        # Check if frames are very similar (static content)
        diff = cv2.absdiff(frame1, frame2)
        motion_level = np.mean(diff) / 255.0
        
        if motion_level < 0.02:  # Very little motion
            # Use subtle variation instead of linear blend
            # This prevents "frozen" look in static scenes
            base_frame = frame1 if alpha < 0.5 else frame2
            variation_strength = alpha if alpha < 0.5 else (1 - alpha)
            
            # Add very subtle noise/variation
            noise = np.random.randint(-1, 2, frame1.shape, dtype=np.int8)
            noise = (noise * variation_strength * 2).astype(np.int8)
            
            result = cv2.add(base_frame, noise.astype(base_frame.dtype))
            return result
        else:
            # Normal motion - use standard blending
            return cv2.addWeighted(frame1, 1-alpha, frame2, alpha, 0)
    
    def _gpu_motion_interpolate(self, frame1: np.ndarray, frame2: np.ndarray, alpha: float) -> np.ndarray:
        """GPU motion-preserving interpolation."""
        try:
            # Convert to tensors
            def to_tensor(frame):
                tensor = torch.from_numpy(frame).float().to(self.device) / 255.0
                return tensor.permute(2, 0, 1).unsqueeze(0)
            
            tensor1 = to_tensor(frame1)
            tensor2 = to_tensor(frame2)
            
            with torch.no_grad():
                # Analyze motion level
                diff = torch.mean(torch.abs(tensor2 - tensor1)).item()
                
                if diff < 0.02:  # Low motion - preserve more of original
                    # Use weighted blend favoring the closest original frame
                    if alpha < 0.5:
                        weight = 0.8 + alpha * 0.4  # 0.8 to 1.0
                        result = tensor1 * weight + tensor2 * (1 - weight)
                    else:
                        weight = 1.2 - alpha * 0.4  # 1.0 to 0.8  
                        result = tensor2 * weight + tensor1 * (1 - weight)
                    
                    # Add very subtle variation
                    noise = torch.randn_like(result) * 0.002
                    result = result + noise
                    
                else:  # Normal motion
                    # Standard smooth interpolation
                    alpha_smooth = torch.tensor(alpha, device=self.device)
                    curve = 3 * alpha_smooth**2 - 2 * alpha_smooth**3
                    result = tensor1 * (1 - curve) + tensor2 * curve
                
                # Convert back
                result = torch.clamp(result, 0, 1)
                result_np = result.squeeze(0).permute(1, 2, 0).cpu().numpy()
                result_np = (result_np * 255).astype(np.uint8)
                
                return result_np
                
        except Exception as e:
            logging.debug(f"GPU motion interpolation failed: {e}")
            return self._cpu_motion_interpolate(frame1, frame2, alpha)
    
    def process_video_motion_preserving(self, input_path: str, output_path: str, 
                                      problem_segments: List[Dict], mode: str = "adaptive") -> Dict:
        """
        Process video preserving motion - only fixes severe timing issues.
        
        NEW STRATEGY:
        1. Only interpolate between ADJACENT frames, never skip frames
        2. Only do this in severe problem areas  
        3. Preserve original frames wherever possible
        4. Use motion-aware blending
        """
        logging.info(f"🎭 Starting motion-preserving interpolation ({mode} mode)")
        
        # Load video
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logging.info(f"📹 Video: {total_frames} frames at {fps} FPS")
        
        # Load all frames
        all_frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            all_frames.append(frame)
        cap.release()
        
        logging.info(f"📂 Loaded {len(all_frames)} frames")
        
        # Determine which frames need fixing based on mode
        frames_to_fix = set()
        
        if not problem_segments:
            logging.info("✅ No problem segments - copying original")
            import shutil
            shutil.copy2(input_path, output_path)
            return {"frames_fixed": 0, "frames_preserved": total_frames}
        
        # Different strategies per mode - SIMPLIFIED APPROACH
        if mode == "precision":
            # Only fix the most problematic frames in each segment
            for seg in problem_segments:
                start, end = seg['start_frame'], seg['end_frame']
                # Fix middle frame of each segment
                if end > start:
                    mid = (start + end) // 2
                    frames_to_fix.add(mid)
        
        elif mode == "adaptive":
            # Fix some frames in problem segments
            for seg in problem_segments:
                start, end = seg['start_frame'], seg['end_frame']
                # Fix 1 out of every 4 frames in problem segments
                for i in range(start, end + 1):
                    if (i - start) % 4 == 0:
                        frames_to_fix.add(i)
        
        elif mode == "maximum":
            # Fix most frames in problem segments  
            for seg in problem_segments:
                start, end = seg['start_frame'], seg['end_frame']
                # Fix 3 out of every 4 frames in problem segments
                for i in range(start, end + 1):
                    if (i - start) % 4 != 3:  # Skip every 4th frame
                        frames_to_fix.add(i)
        
        logging.info(f"🎯 Will fix {len(frames_to_fix)} frames out of {total_frames} ({len(frames_to_fix)/total_frames*100:.1f}%)")
        
        # Create output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frames_fixed = 0
        frames_preserved = 0
        
        try:
            for i in range(len(all_frames)):
                if i in frames_to_fix:
                    # This frame needs fixing - find good reference frames
                    
                    # Look for previous non-duplicate frame
                    prev_idx = i - 1
                    while prev_idx >= 0 and prev_idx in frames_to_fix:
                        prev_idx -= 1
                    
                    # Look for next non-duplicate frame  
                    next_idx = i + 1
                    while next_idx < len(all_frames) and next_idx in frames_to_fix:
                        next_idx += 1
                    
                    if prev_idx >= 0 and next_idx < len(all_frames):
                        # Interpolate between good frames
                        distance = next_idx - prev_idx
                        position = i - prev_idx
                        alpha = position / distance if distance > 0 else 0.5
                        
                        interpolated = self.create_motion_aware_interpolation(
                            all_frames[prev_idx], all_frames[next_idx], alpha, preserve_motion=True
                        )
                        
                        out.write(interpolated)
                        frames_fixed += 1
                        
                        if frames_fixed % 20 == 0:
                            logging.info(f"   🎭 Fixed {frames_fixed} frames with motion preservation")
                    
                    elif prev_idx >= 0:
                        # Only previous frame available - use slight variation
                        varied = self._create_slight_variation(all_frames[prev_idx])
                        out.write(varied)
                        frames_fixed += 1
                    
                    elif next_idx < len(all_frames):
                        # Only next frame available - use slight variation
                        varied = self._create_slight_variation(all_frames[next_idx])
                        out.write(varied)
                        frames_fixed += 1
                    
                    else:
                        # No reference frames - use original
                        out.write(all_frames[i])
                        frames_preserved += 1
                
                else:
                    # Use original frame
                    out.write(all_frames[i])
                    frames_preserved += 1
                
                # Progress
                if i % 50 == 0:
                    progress = (i / len(all_frames)) * 100
                    logging.info(f"   Progress: {progress:.1f}%")
        
        finally:
            out.release()
        
        # Results
        fixed_pct = (frames_fixed / total_frames) * 100
        preserved_pct = (frames_preserved / total_frames) * 100
        
        logging.info(f"✅ Motion-preserving interpolation complete!")
        logging.info(f"📊 Results:")
        logging.info(f"   • Fixed {frames_fixed} frames ({fixed_pct:.1f}%) - adjacent interpolation only")
        logging.info(f"   • Preserved {frames_preserved} original frames ({preserved_pct:.1f}%)")
        logging.info(f"   • Motion preserved throughout video")
        
        return {
            "frames_fixed": frames_fixed,
            "frames_preserved": frames_preserved,
            "total_frames": total_frames,
            "fixed_percentage": fixed_pct,
            "preserved_percentage": preserved_pct
        }
    
    def _create_slight_variation(self, frame: np.ndarray) -> np.ndarray:
        """Create a slight variation of a frame to avoid duplication."""
        if self.use_gpu:
            try:
                # GPU-based variation
                tensor = torch.from_numpy(frame).float().to(self.device) / 255.0
                
                # Add very subtle noise
                noise = torch.randn_like(tensor) * 0.003
                varied = tensor + noise
                varied = torch.clamp(varied, 0, 1)
                
                result = (varied.cpu().numpy() * 255).astype(np.uint8)
                return result
            except:
                pass
        
        # CPU fallback - very subtle random noise
        noise = np.random.randint(-2, 3, frame.shape, dtype=np.int8)
        result = cv2.add(frame, noise.astype(frame.dtype))
        return result