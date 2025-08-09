"""
AI-powered freeze repair using RIFE interpolation
"""
import cv2
import numpy as np
import logging
import torch
from .timecode_freeze_predictor import predict_freezes_from_timecodes

def repair_freezes_with_rife(video_path, freeze_predictions, output_path, rife_model):
    """
    Точечное исправление замороженных кадров с помощью RIFE.
    Заменяет только дубликаты на AI-интерполированные кадры.
    """
    logging.info("🤖 Starting AI freeze repair with RIFE...")
    
    if not freeze_predictions:
        logging.info("No freezes to repair - copying original video")
        import shutil
        shutil.copy2(video_path, output_path)
        return True
    
    # Create set of frames that need repair
    frames_to_repair = set()
    for segment in freeze_predictions:
        for pred in segment['predictions']:
            frames_to_repair.add(pred['frame'])
    
    logging.info(f"Will repair {len(frames_to_repair)} frozen frames")
    
    # Open input video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Load all frames
    all_frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
        frame_idx += 1
    cap.release()
    
    logging.info(f"Loaded {len(all_frames)} frames")
    
    # Create output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    repaired_count = 0
    
    # Process each frame
    for frame_idx in range(len(all_frames)):
        current_frame = all_frames[frame_idx]
        
        if frame_idx in frames_to_repair:
            # This frame needs AI repair
            prev_frame, next_frame = find_neighbor_frames(all_frames, frame_idx, frames_to_repair)
            
            if prev_frame is not None and next_frame is not None:
                # Use RIFE to interpolate
                try:
                    interpolated = interpolate_with_rife(prev_frame, next_frame, rife_model)
                    if interpolated is not None:
                        current_frame = interpolated
                        repaired_count += 1
                        logging.info(f"   Repaired frame {frame_idx} using RIFE")
                    else:
                        logging.warning(f"   RIFE failed for frame {frame_idx}, using original")
                except Exception as e:
                    logging.warning(f"   RIFE error for frame {frame_idx}: {e}")
            else:
                logging.warning(f"   Cannot find neighbors for frame {frame_idx}")
        
        out.write(current_frame)
        
        if frame_idx % 100 == 0:
            logging.info(f"   Progress: {frame_idx}/{total_frames} ({repaired_count} repaired)")
    
    out.release()
    
    repair_pct = (repaired_count / len(frames_to_repair) * 100) if frames_to_repair else 0
    
    logging.info(f"✅ AI repair complete!")
    logging.info(f"   Repaired: {repaired_count}/{len(frames_to_repair)} frames ({repair_pct:.1f}%)")
    
    return True

def find_neighbor_frames(all_frames, target_idx, frozen_frames):
    """
    Найти ближайшие незамороженные кадры до и после целевого кадра.
    """
    prev_frame = None
    next_frame = None
    
    # Поиск предыдущего незамороженного кадра
    for i in range(target_idx - 1, -1, -1):
        if i not in frozen_frames:
            prev_frame = all_frames[i]
            break
    
    # Поиск следующего незамороженного кадра  
    for i in range(target_idx + 1, len(all_frames)):
        if i not in frozen_frames:
            next_frame = all_frames[i]
            break
    
    return prev_frame, next_frame

def interpolate_with_rife(prev_frame, next_frame, rife_model):
    """
    Интерполировать один кадр между двумя с помощью RIFE.
    """
    try:
        if not rife_model or not rife_model.available:
            logging.warning("RIFE model not available")
            return None
        
        # Convert frames to tensors
        prev_tensor = frame_to_tensor(prev_frame, rife_model.device)
        next_tensor = frame_to_tensor(next_frame, rife_model.device)
        
        # Interpolate middle frame (timestep=0.5)
        with torch.no_grad():
            interpolated_tensor = rife_model.interpolate(prev_tensor, next_tensor, 0.5)
            
        # Convert back to frame
        interpolated_frame = tensor_to_frame(interpolated_tensor)
        
        return interpolated_frame
        
    except Exception as e:
        logging.error(f"RIFE interpolation error: {e}")
        return None

def frame_to_tensor(frame, device):
    """Convert OpenCV frame to RIFE tensor format."""
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    frame_float = frame_rgb.astype(np.float32) / 255.0
    
    # Convert to tensor (C, H, W)
    tensor = torch.from_numpy(frame_float.transpose(2, 0, 1)).unsqueeze(0).to(device)
    
    return tensor

def tensor_to_frame(tensor):
    """Convert RIFE tensor back to OpenCV frame."""
    # Convert to numpy (H, W, C)
    frame_np = tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    
    # Denormalize to [0, 255]
    frame_np = (frame_np * 255.0).clip(0, 255).astype(np.uint8)
    
    # Convert RGB back to BGR for OpenCV
    frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
    
    return frame_bgr

def create_ai_repair_report(repaired_count, total_freezes):
    """Create detailed repair report."""
    if total_freezes == 0:
        return "🤖 No freezes detected - no AI repair needed!"
    
    repair_pct = (repaired_count / total_freezes * 100) if total_freezes > 0 else 0
    
    report = f"""🤖 AI FREEZE REPAIR COMPLETE!

📊 Repair Results:
• Detected freezes: {total_freezes}
• Successfully repaired: {repaired_count} ({repair_pct:.1f}%)
• Failed repairs: {total_freezes - repaired_count}

⚡ Method: RIFE point interpolation
🎯 Strategy: Replace frozen frames with AI-generated intermediate frames
🔧 Neighbors: Uses closest non-frozen frames for interpolation

✨ Result: Smooth video with AI-repaired freeze points!"""
    
    return report