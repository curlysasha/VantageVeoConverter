"""
Smart frame interpolator - only fixes timing issues, preserves good frames
"""
import cv2
import numpy as np
import torch
import logging
from typing import List, Dict, Tuple

class SmartFrameInterpolator:
    """Smart interpolator that only fixes actual timing problems."""
    
    def __init__(self, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_gpu = torch.cuda.is_available() and device == "cuda"
        
        logging.info(f"🧠 Smart interpolator initialized on {self.device}")
    
    def analyze_timing_issues(self, timecode_path: str, fps: float, 
                            mode_threshold: float = 0.15) -> List[Dict]:
        """
        Analyze timecode file to find actual timing problems.
        Only returns segments that have real timing issues.
        """
        with open(timecode_path, 'r') as f:
            lines = [line.strip() for line in f.readlines() if not line.startswith('#')]
        
        timestamps = [int(line) for line in lines if line.isdigit()]
        if len(timestamps) < 10:
            return []
        
        expected_interval = 1000.0 / fps  # Expected interval in ms
        problem_segments = []
        
        logging.info(f"🔍 Analyzing timing with {len(timestamps)} timestamps")
        logging.info(f"📏 Expected frame interval: {expected_interval:.1f}ms")
        
        # Find consecutive problematic areas
        current_problem = None
        
        for i in range(1, len(timestamps)):
            actual_interval = timestamps[i] - timestamps[i-1]
            
            if actual_interval <= 0:
                # Invalid timestamp - definite problem
                deviation = 1.0  # Max deviation
            else:
                speed_ratio = expected_interval / actual_interval
                deviation = abs(speed_ratio - 1.0)
            
            is_problem = deviation > mode_threshold
            
            if is_problem:
                if current_problem is None:
                    # Start new problem segment
                    current_problem = {
                        'start_frame': max(0, i - 2),  # Include some context
                        'end_frame': i + 2,
                        'issues': [],
                        'max_deviation': deviation
                    }
                else:
                    # Extend current problem
                    current_problem['end_frame'] = i + 2
                    current_problem['max_deviation'] = max(current_problem['max_deviation'], deviation)
                
                current_problem['issues'].append({
                    'frame': i,
                    'deviation': deviation,
                    'actual_interval': actual_interval,
                    'expected_interval': expected_interval
                })
            
            else:
                # No problem at this frame
                if current_problem is not None:
                    # End current problem segment
                    current_problem['end_frame'] = min(len(timestamps), current_problem['end_frame'])
                    
                    # Only add if it's a significant problem
                    if len(current_problem['issues']) >= 2 or current_problem['max_deviation'] > 0.3:
                        problem_segments.append(current_problem)
                    
                    current_problem = None
        
        # Add final segment if exists
        if current_problem is not None:
            current_problem['end_frame'] = min(len(timestamps), current_problem['end_frame'])
            if len(current_problem['issues']) >= 2 or current_problem['max_deviation'] > 0.3:
                problem_segments.append(current_problem)
        
        # Log results
        total_problem_frames = sum(seg['end_frame'] - seg['start_frame'] for seg in problem_segments)
        problem_pct = (total_problem_frames / len(timestamps)) * 100
        
        logging.info(f"📊 Timing analysis results:")
        logging.info(f"   • Found {len(problem_segments)} timing problem areas")
        logging.info(f"   • Affects {total_problem_frames}/{len(timestamps)} frames ({problem_pct:.1f}%)")
        
        for i, seg in enumerate(problem_segments):
            seg_size = seg['end_frame'] - seg['start_frame']
            logging.info(f"   Problem {i+1}: frames {seg['start_frame']}-{seg['end_frame']} "
                        f"({seg_size} frames, max_dev={seg['max_deviation']:.3f})")
        
        return problem_segments
    
    def create_smart_interpolation(self, frame1: np.ndarray, frame2: np.ndarray, 
                                  alpha: float = 0.5) -> np.ndarray:
        """Create interpolated frame with smart GPU/CPU selection."""
        if self.use_gpu:
            return self._gpu_interpolate(frame1, frame2, alpha)
        else:
            return self._cpu_interpolate(frame1, frame2, alpha)
    
    def _cpu_interpolate(self, frame1: np.ndarray, frame2: np.ndarray, alpha: float) -> np.ndarray:
        """Simple CPU interpolation."""
        return cv2.addWeighted(frame1, 1-alpha, frame2, alpha, 0)
    
    def _gpu_interpolate(self, frame1: np.ndarray, frame2: np.ndarray, alpha: float) -> np.ndarray:
        """GPU-accelerated interpolation."""
        try:
            # Convert to tensors
            def to_tensor(frame):
                tensor = torch.from_numpy(frame).float().to(self.device) / 255.0
                return tensor.permute(2, 0, 1).unsqueeze(0)
            
            tensor1 = to_tensor(frame1)
            tensor2 = to_tensor(frame2)
            
            with torch.no_grad():
                # Smooth interpolation curve
                alpha_smooth = torch.tensor(alpha, device=self.device)
                curve = 3 * alpha_smooth**2 - 2 * alpha_smooth**3
                
                # Interpolate
                result = tensor1 * (1 - curve) + tensor2 * curve
                
                # Convert back
                result = torch.clamp(result, 0, 1)
                result_np = result.squeeze(0).permute(1, 2, 0).cpu().numpy()
                result_np = (result_np * 255).astype(np.uint8)
                
                return result_np
                
        except Exception as e:
            logging.debug(f"GPU interpolation failed: {e}, using CPU")
            return self._cpu_interpolate(frame1, frame2, alpha)
    
    def process_video_smart(self, input_path: str, output_path: str, 
                          timecode_path: str, fps: float, 
                          mode: str = "adaptive") -> Dict:
        """
        Smart video processing: only fix actual timing problems.
        
        Strategy:
        1. Analyze timecode for real timing issues
        2. Load only affected frames + context
        3. Fix only problematic segments
        4. Preserve all good frames unchanged
        """
        logging.info(f"🧠 Starting smart interpolation ({mode} mode)")
        
        # Different thresholds for different modes
        if mode == "precision":
            threshold = 0.05  # Very conservative
        elif mode == "adaptive": 
            threshold = 0.15  # Moderate
        elif mode == "maximum":
            threshold = 0.25  # More aggressive
        else:
            threshold = 0.15
        
        # Analyze timing issues
        problem_segments = self.analyze_timing_issues(timecode_path, fps, threshold)
        
        if not problem_segments:
            logging.info("✅ No timing issues found - video is already good!")
            import shutil
            shutil.copy2(input_path, output_path)
            return {"problem_segments": 0, "frames_fixed": 0, "frames_preserved": "all"}
        
        # Load video info
        cap = cv2.VideoCapture(input_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        logging.info(f"📹 Video: {total_frames} frames, {video_fps} FPS, {width}x{height}")
        
        # Process video frame by frame
        cap = cv2.VideoCapture(input_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, video_fps, (width, height))
        
        frames_fixed = 0
        frames_preserved = 0
        
        try:
            logging.info("🎬 Processing video with smart interpolation...")
            
            frame_buffer = {}  # Cache frames we need for interpolation
            
            for frame_idx in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Check if this frame is in a problem segment
                in_problem_segment = any(
                    seg['start_frame'] <= frame_idx <= seg['end_frame']
                    for seg in problem_segments
                )
                
                if in_problem_segment:
                    # This frame needs special handling
                    
                    # Find the problem segment
                    current_segment = None
                    for seg in problem_segments:
                        if seg['start_frame'] <= frame_idx <= seg['end_frame']:
                            current_segment = seg
                            break
                    
                    if current_segment:
                        # Get reference frames for interpolation
                        ref_start = max(0, current_segment['start_frame'] - 1)
                        ref_end = min(total_frames - 1, current_segment['end_frame'] + 1)
                        
                        # Load reference frames if not cached
                        if ref_start not in frame_buffer:
                            cap_ref = cv2.VideoCapture(input_path)
                            cap_ref.set(cv2.CAP_PROP_POS_FRAMES, ref_start)
                            ret_ref, ref_frame = cap_ref.read()
                            if ret_ref:
                                frame_buffer[ref_start] = ref_frame
                            cap_ref.release()
                        
                        if ref_end not in frame_buffer:
                            cap_ref = cv2.VideoCapture(input_path)
                            cap_ref.set(cv2.CAP_PROP_POS_FRAMES, ref_end)
                            ret_ref, ref_frame = cap_ref.read()
                            if ret_ref:
                                frame_buffer[ref_end] = ref_frame
                            cap_ref.release()
                        
                        # Create interpolated frame
                        if ref_start in frame_buffer and ref_end in frame_buffer:
                            # Calculate interpolation position
                            segment_length = ref_end - ref_start
                            position = frame_idx - ref_start
                            alpha = position / segment_length if segment_length > 0 else 0.5
                            
                            # Create interpolated frame
                            interpolated = self.create_smart_interpolation(
                                frame_buffer[ref_start], 
                                frame_buffer[ref_end], 
                                alpha
                            )
                            out.write(interpolated)
                            frames_fixed += 1
                            
                            if frames_fixed % 10 == 0:
                                logging.info(f"   🔧 Fixed {frames_fixed} problematic frames")
                        else:
                            # Fallback to original frame
                            out.write(frame)
                            frames_preserved += 1
                    else:
                        # Shouldn't happen, but fallback
                        out.write(frame)
                        frames_preserved += 1
                
                else:
                    # Frame is good - preserve as-is
                    out.write(frame)
                    frames_preserved += 1
                
                # Progress logging
                if frame_idx % 50 == 0:
                    progress = (frame_idx / total_frames) * 100
                    logging.info(f"   Progress: {progress:.1f}%")
        
        finally:
            cap.release()
            out.release()
        
        # Results
        fixed_pct = (frames_fixed / total_frames) * 100
        preserved_pct = (frames_preserved / total_frames) * 100
        
        logging.info(f"✅ Smart interpolation complete!")
        logging.info(f"📊 Results:")
        logging.info(f"   • Fixed {frames_fixed} problematic frames ({fixed_pct:.1f}%)")
        logging.info(f"   • Preserved {frames_preserved} good frames ({preserved_pct:.1f}%)")
        logging.info(f"   • Processed {len(problem_segments)} problem segments")
        
        return {
            "problem_segments": len(problem_segments),
            "frames_fixed": frames_fixed,
            "frames_preserved": frames_preserved,
            "total_frames": total_frames,
            "fixed_percentage": fixed_pct,
            "preserved_percentage": preserved_pct
        }