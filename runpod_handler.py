"""
VantageVeoConverter RunPod Serverless Handler
Полноценный serverless handler для синхронизации видео с аудио + RIFE AI интерполяция
"""
import runpod
import tempfile
import os
import json
import logging
import shutil
import sys
import subprocess
from pathlib import Path
import requests
import time
import base64

# Импорты существующих модулей с обработкой возможных ошибок
try:
    from src.comfy_rife import ComfyRIFE
    from src.timing_analyzer import analyze_timing_changes
    from src.ai_freeze_repair import repair_freezes_with_rife
    from src.timecode_freeze_predictor import predict_freezes_from_timecodes
    from src.audio_sync import *
    from src.physical_retime import create_physical_retime
    from src.triple_diagnostic import create_triple_diagnostic
    from src.binary_utils import check_all_binaries, get_ffmpeg, get_ffprobe
except ImportError as e:
    logger.warning(f"Some modules failed to import: {e}")
    logger.warning("Some features may be unavailable")

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Добавляем bin директорию в PATH
bin_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin")
if os.path.exists(bin_dir):
    current_path = os.environ.get('PATH', '')
    os.environ['PATH'] = f"{bin_dir}{os.pathsep}{current_path}"
    logger.info(f"Added {bin_dir} to PATH")

# --- Global Setup: Load Models Once ---
import torch
try:
    import whisper
except ImportError as e:
    logger.warning(f"Whisper import failed: {e}")
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "numba"], 
                  capture_output=True, text=True)
    import whisper

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WHISPER_MODEL = None
RIFE_MODEL = None

def initialize_models():
    """Initialize Whisper and RIFE models once at container startup."""
    global WHISPER_MODEL, RIFE_MODEL
    
    logger.info("🚀 Initializing models...")
    
    # Load Whisper
    try:
        logger.info(f"Loading Whisper model ('base') to {DEVICE}...")
        WHISPER_MODEL = whisper.load_model("base", device=DEVICE) 
        logger.info("✅ Whisper model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load Whisper model: {e}")
        WHISPER_MODEL = None
    
    # Initialize RIFE
    try:
        logger.info(f"Initializing RIFE model on {DEVICE}...")
        RIFE_MODEL = ComfyRIFE(DEVICE)
        logger.info("✅ RIFE model initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize RIFE model: {e}")
        RIFE_MODEL = None

def check_dependencies():
    """Check if required external tools are available."""
    try:
        missing = check_all_binaries()
        if missing:
            logger.error(f"❌ Missing binary dependencies: {', '.join(missing)}")
            return False
        logger.info("✅ All binary dependencies available")
        return True
    except Exception as e:
        logger.error(f"❌ Error checking dependencies: {e}")
        return False

def download_file(url, filename=None):
    """Download file from URL to temporary location."""
    try:
        if not filename:
            # Generate filename from URL
            filename = url.split('/')[-1]
            if not filename or '.' not in filename:
                # Generate based on content type
                response = requests.head(url, timeout=30)
                content_type = response.headers.get('content-type', '')
                if 'video' in content_type:
                    filename = f"input_video_{int(time.time())}.mp4"
                elif 'audio' in content_type:
                    filename = f"input_audio_{int(time.time())}.wav"
                else:
                    filename = f"input_file_{int(time.time())}"
        
        # Create temporary file
        temp_dir = tempfile.gettempdir()
        local_path = os.path.join(temp_dir, filename)
        
        logger.info(f"📥 Downloading {url} to {local_path}")
        
        # Download file
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        logger.info(f"✅ Downloaded {os.path.getsize(local_path)} bytes")
        return local_path
        
    except Exception as e:
        logger.error(f"❌ Failed to download {url}: {e}")
        raise Exception(f"Download failed: {str(e)}")

def upload_result_file(file_path, job_id):
    """Upload result file and return URL/base64."""
    try:
        if not os.path.exists(file_path):
            return None
            
        file_size = os.path.getsize(file_path)
        logger.info(f"📤 Uploading result file: {file_size} bytes")
        
        # For files < 10MB, return as base64
        if file_size < 10 * 1024 * 1024:
            with open(file_path, 'rb') as f:
                encoded = base64.b64encode(f.read()).decode('utf-8')
            logger.info("✅ File encoded as base64")
            return f"data:video/mp4;base64,{encoded}"
        
        # For larger files, try to upload to RunPod's storage
        try:
            from runpod.serverless.utils import rp_upload
            url = rp_upload.upload_image(job_id, file_path)
            logger.info(f"✅ File uploaded to: {url}")
            return url
        except Exception as upload_error:
            logger.warning(f"Upload failed, falling back to base64: {upload_error}")
            # Fallback to base64 even for large files
            with open(file_path, 'rb') as f:
                encoded = base64.b64encode(f.read()).decode('utf-8')
            return f"data:video/mp4;base64,{encoded}"
            
    except Exception as e:
        logger.error(f"❌ Failed to process result file: {e}")
        return None

def handler(job):
    """
    Main RunPod serverless handler for VantageVeoConverter.
    
    Input format:
    {
        "input": {
            "video_url": "https://example.com/video.mp4",
            "audio_url": "https://example.com/audio.wav", 
            "use_rife": true,  # Optional, default True
            "diagnostic_mode": false,  # Optional, default False
            "rife_mode": "adaptive",  # Optional: "off", "adaptive", "precision", "maximum"
        }
    }
    """
    job_id = job.get("id", "unknown")
    logger.info(f"🎬 Starting VantageVeoConverter job {job_id}")
    
    try:
        # Parse input
        job_input = job.get("input", {})
        video_url = job_input.get("video_url")
        audio_url = job_input.get("audio_url")
        use_rife = job_input.get("use_rife", True)
        diagnostic_mode = job_input.get("diagnostic_mode", False)
        rife_mode = job_input.get("rife_mode", "adaptive")
        
        # Validate input
        if not video_url:
            return {"error": "video_url is required"}
        if not audio_url:
            return {"error": "audio_url is required"}
        
        logger.info(f"📋 Job config: use_rife={use_rife}, diagnostic={diagnostic_mode}, rife_mode={rife_mode}")
        
        # Check dependencies
        if not check_dependencies():
            return {"error": "Missing required binary dependencies (ffmpeg, mp4fpsmod)"}
        
        # Check models
        if WHISPER_MODEL is None:
            return {"error": "Whisper model not loaded"}
        if use_rife and RIFE_MODEL is None:
            return {"error": "RIFE model not loaded but use_rife=True"}
        
        # Download files
        logger.info("📥 Downloading input files...")
        try:
            video_path = download_file(video_url, f"input_video_{job_id}.mp4")
            audio_path = download_file(audio_url, f"input_audio_{job_id}.wav")
        except Exception as e:
            return {"error": f"Failed to download input files: {str(e)}"}
        
        # Create temporary directory for outputs
        temp_output_dir = tempfile.mkdtemp(prefix=f"vantage_job_{job_id}_")
        logger.info(f"📁 Working directory: {temp_output_dir}")
        
        try:
            if diagnostic_mode:
                # Diagnostic mode workflow
                logger.info("🔍 Running diagnostic mode...")
                result = diagnostic_workflow(video_path, audio_path, temp_output_dir, job_id)
            else:
                # Normal synchronization workflow  
                logger.info("⚙️ Running synchronization workflow...")
                result = synchronization_workflow(
                    video_path, 
                    audio_path, 
                    temp_output_dir, 
                    use_rife, 
                    rife_mode, 
                    job_id
                )
            
            # Upload result files
            if "synchronized_video" in result:
                result["synchronized_video_url"] = upload_result_file(result["synchronized_video"], job_id)
            
            if "diagnostic_video" in result:
                result["diagnostic_video_url"] = upload_result_file(result["diagnostic_video"], job_id)
            
            if "timecodes" in result:
                # For timecodes, read as text
                try:
                    with open(result["timecodes"], 'r') as f:
                        result["timecodes_content"] = f.read()
                except:
                    pass
            
            # Clean up local files but keep URLs
            cleanup_paths = ["synchronized_video", "diagnostic_video", "timecodes"]
            for path_key in cleanup_paths:
                if path_key in result:
                    del result[path_key]
            
            logger.info(f"✅ Job {job_id} completed successfully")
            return {
                "success": True,
                "job_id": job_id,
                "processing_time": result.get("processing_time", 0),
                **result
            }
            
        finally:
            # Clean up temporary directory
            try:
                shutil.rmtree(temp_output_dir)
                logger.info(f"🧹 Cleaned up {temp_output_dir}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to clean up {temp_output_dir}: {e}")
    
    except Exception as e:
        logger.error(f"❌ Job {job_id} failed: {str(e)}", exc_info=True)
        return {
            "error": str(e),
            "job_id": job_id,
            "success": False
        }
    
    finally:
        # Clean up downloaded files
        try:
            if 'video_path' in locals() and os.path.exists(video_path):
                os.unlink(video_path)
            if 'audio_path' in locals() and os.path.exists(audio_path):
                os.unlink(audio_path)
        except Exception as e:
            logger.warning(f"⚠️ Failed to clean up downloaded files: {e}")

def synchronization_workflow(video_path, audio_path, output_dir, use_rife, rife_mode, job_id):
    """Main synchronization workflow."""
    start_time = time.time()
    
    # Setup paths
    paths = {
        "original_video": video_path,
        "target_audio": audio_path,
        "sync_audio": os.path.join(output_dir, "sync_audio.wav"),
        "timecodes": os.path.join(output_dir, "timecodes.txt"),
        "retimed_video": os.path.join(output_dir, "retimed_video.mp4"),
        "final_output": os.path.join(output_dir, "synchronized_video.mp4")
    }
    
    logger.info("🎵 Step 1: Audio synchronization...")
    
    # Audio synchronization
    try:
        sync_result = synchronize_audio_video_workflow(
            video_path=paths["original_video"],
            target_audio_path=paths["target_audio"],
            whisper_model=WHISPER_MODEL,
            output_dir=output_dir
        )
        
        if not sync_result["success"]:
            raise Exception(f"Audio sync failed: {sync_result.get('error', 'Unknown error')}")
        
        paths["sync_audio"] = sync_result["sync_audio_path"]
        paths["timecodes"] = sync_result["timecodes_path"]
        
    except Exception as e:
        raise Exception(f"Audio synchronization failed: {str(e)}")
    
    logger.info("🎬 Step 2: Video retiming...")
    
    # Video retiming
    try:
        # Create physical retime for CFR video
        create_physical_retime(
            paths["original_video"],
            paths["timecodes"], 
            paths["retimed_video"]
        )
        
        # Add audio
        mux_final_output(
            paths["retimed_video"],
            paths["sync_audio"],
            paths["final_output"]
        )
        
    except Exception as e:
        raise Exception(f"Video retiming failed: {str(e)}")
    
    # Apply RIFE if requested
    if use_rife and rife_mode != "off" and RIFE_MODEL is not None:
        logger.info(f"🤖 Step 3: RIFE AI repair (mode: {rife_mode})...")
        
        try:
            # Predict freezes
            freeze_data = predict_freezes_from_timecodes(paths["timecodes"])
            
            if freeze_data and len(freeze_data) > 0:
                # Apply RIFE repair
                rife_output_path = os.path.join(output_dir, "rife_repaired.mp4")
                repair_result = repair_freezes_with_rife(
                    paths["final_output"],
                    freeze_data,
                    rife_output_path,
                    RIFE_MODEL,
                    mode=rife_mode
                )
                
                if repair_result["success"]:
                    # Replace final output with RIFE result
                    shutil.move(rife_output_path, paths["final_output"])
                    logger.info(f"✅ RIFE repair applied: {repair_result['repairs_applied']} repairs")
                else:
                    logger.warning(f"⚠️ RIFE repair failed: {repair_result.get('error', 'Unknown error')}")
            else:
                logger.info("ℹ️ No freezes detected, skipping RIFE repair")
                
        except Exception as e:
            logger.warning(f"⚠️ RIFE processing failed, continuing without repair: {str(e)}")
    
    processing_time = time.time() - start_time
    logger.info(f"✅ Synchronization completed in {processing_time:.2f}s")
    
    return {
        "synchronized_video": paths["final_output"],
        "timecodes": paths["timecodes"],
        "processing_time": processing_time,
        "use_rife": use_rife,
        "rife_mode": rife_mode if use_rife else None
    }

def diagnostic_workflow(video_path, audio_path, output_dir, job_id):
    """Diagnostic workflow with visual freeze detection."""
    start_time = time.time()
    
    logger.info("🔍 Running diagnostic analysis...")
    
    # First run normal sync workflow
    sync_result = synchronization_workflow(
        video_path, audio_path, output_dir, 
        use_rife=True, rife_mode="adaptive", job_id=job_id
    )
    
    # Create diagnostic visualization
    diagnostic_path = os.path.join(output_dir, "diagnostic_comparison.mp4")
    
    try:
        # Create triple diagnostic (Original + Detection + AI Repair)
        create_triple_diagnostic(
            original_video=video_path,
            synchronized_video=sync_result["synchronized_video"], 
            timecodes_file=sync_result["timecodes"],
            output_path=diagnostic_path
        )
        
        processing_time = time.time() - start_time
        logger.info(f"✅ Diagnostic completed in {processing_time:.2f}s")
        
        return {
            **sync_result,
            "diagnostic_video": diagnostic_path,
            "processing_time": processing_time,
            "diagnostic_mode": True
        }
        
    except Exception as e:
        logger.warning(f"⚠️ Diagnostic visualization failed: {str(e)}")
        # Return sync result without diagnostic
        return {
            **sync_result,
            "diagnostic_error": str(e)
        }

# Initialize models when module is loaded
logger.info("🚀 VantageVeoConverter RunPod Handler starting...")
initialize_models()

# Start RunPod serverless worker
if __name__ == "__main__":
    logger.info("🟢 Starting RunPod serverless worker...")
    runpod.serverless.start({
        "handler": handler,
        "return_aggregate_stream": False,
        "concurrency_modifier": lambda current: min(current, 3)  # Max 3 concurrent jobs
    })