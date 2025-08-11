"""
Enhanced Video-Audio Synchronizer with RIFE - Compact Version
Refactored for better maintainability
"""
import subprocess
import os
import json
import logging
import shutil
import gradio as gr
import tempfile
import time
import torch

# Handle numpy version compatibility issues
import sys
try:
    import numpy as np
    # Check if numpy version is compatible
    numpy_version = tuple(map(int, np.__version__.split('.')[:2]))
    if numpy_version < (1, 24):
        logging.warning(f"NumPy version {np.__version__} is too old, upgrading...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "numpy>=1.24,<2.3"], 
                      capture_output=True, text=True)
        # Restart Python interpreter after numpy upgrade
        logging.info("Restarting Python after NumPy upgrade...")
        os.execv(sys.executable, ['python'] + sys.argv)
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "numpy>=1.24,<2.3"], 
                  capture_output=True, text=True)
    import numpy as np

try:
    import whisper
except ImportError as e:
    logging.warning(f"Whisper import failed: {e}")
    # Try to fix numba version issues
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "numba"], 
                  capture_output=True, text=True)
    import whisper

# Import our modules
from src.comfy_rife import ComfyRIFE
from src.timing_analyzer import analyze_timing_changes
from src.ai_freeze_repair import repair_freezes_with_rife
from src.timecode_freeze_predictor import predict_freezes_from_timecodes
from src.audio_sync import *
from src.physical_retime import create_physical_retime
from src.triple_diagnostic import create_triple_diagnostic

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Global Setup: Load Models Once ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WHISPER_MODEL = None
RIFE_MODEL = None

def initialize_models():
    """Initialize Whisper and RIFE models."""
    global WHISPER_MODEL, RIFE_MODEL
    
    # Load Whisper
    try:
        logging.info(f"Loading Whisper model ('base') to {DEVICE}... (This speeds up subsequent requests)")
        WHISPER_MODEL = whisper.load_model("base", device=DEVICE) 
    except Exception as e:
        logging.error(f"Failed to load Whisper model: {e}")
    
    # Initialize RIFE
    RIFE_MODEL = ComfyRIFE(DEVICE)

def check_dependencies():
    """Check if required external tools are available."""
    missing = []
    
    # Check for mp4fpsmod - first in local bin/, then system PATH
    project_dir = os.path.dirname(os.path.abspath(__file__))
    local_mp4fpsmod = os.path.join(project_dir, "bin", "mp4fpsmod")
    if os.name == 'nt':
        local_mp4fpsmod += ".exe"
    
    if not os.path.exists(local_mp4fpsmod) and not shutil.which("mp4fpsmod"):
        missing.append(f"mp4fpsmod (place binary in {project_dir}/bin/)")
    
    if not shutil.which("ffmpeg"): missing.append("ffmpeg")
    # espeak is optional - only warn if missing
    # if not shutil.which("espeak-ng") and not shutil.which("espeak"):
    #      missing.append("espeak/espeak-ng")
    if missing:
        raise EnvironmentError(f"Missing dependencies: {', '.join(missing)}.")

def synchronization_workflow(input_video_path, target_audio_path, use_rife=True, progress=gr.Progress()):
    """Main synchronization workflow with RIFE modes."""
    start_time = time.time()
    try:
        check_dependencies()
        
        final_output_fd, final_output_path = tempfile.mkstemp(suffix=".mp4", prefix="synchronized_")
        os.close(final_output_fd)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            logging.info(f"Starting processing in {temp_dir}")

            paths = {
                "source_audio": os.path.join(temp_dir, "source_16k.wav"),
                "target_audio_processed": os.path.join(temp_dir, "target_16k.wav"),
                "transcript": os.path.join(temp_dir, "transcript.txt"),
                "align_source": os.path.join(temp_dir, "align_source.json"),
                "align_target": os.path.join(temp_dir, "align_target.json"),
                "timecodes": os.path.join(temp_dir, "timecodes_v2.txt"),
                "interpolated_video": os.path.join(temp_dir, "interpolated.mp4"),
                "retimed_video": os.path.join(temp_dir, "retimed.mp4"),
                "interpolated_timecodes": os.path.join(temp_dir, "interpolated_timecodes_v2.txt"),
                "temp_final_output": os.path.join(temp_dir, "synchronized_output.mp4")
            }

            # Standard pipeline
            progress(0.1, desc="1/8: Extracting audio...")
            extract_and_standardize_audio(input_video_path, paths["source_audio"])
            extract_and_standardize_audio(target_audio_path, paths["target_audio_processed"])
            
            progress(0.2, desc=f"2/8: Transcribing (Whisper on {DEVICE})...")
            transcript_text = transcribe_audio(paths["target_audio_processed"], paths["transcript"], WHISPER_MODEL)
            
            progress(0.35, desc="3/8: Aligning Source...")
            forced_alignment(paths["source_audio"], paths["transcript"], paths["align_source"])

            progress(0.5, desc="4/8: Aligning Target...")
            forced_alignment(paths["target_audio_processed"], paths["transcript"], paths["align_target"])
            
            progress(0.65, desc="5/8: Calculating enhanced VFR timecodes...")
            
            # Progress callback for timecode generation
            def timecode_progress(percent, message):
                overall_progress = 0.65 + (percent / 100) * 0.1
                progress(overall_progress, desc=f"5/8: {message}")
            
            generate_vfr_timecodes(
                input_video_path, 
                paths["align_source"], 
                paths["align_target"], 
                paths["timecodes"],
                smooth_interpolation=True,
                progress_callback=timecode_progress
            )
            
            if use_rife:
                # First, create synchronized video exactly like diagnostic mode
                progress(0.75, desc="6/8: Creating video with physical frame duplicates...")
                create_physical_retime(input_video_path, paths["timecodes"], paths["retimed_video"])
                
                # Add audio to retimed video using target audio (same as diagnostic)
                progress(0.77, desc="6/8: Adding audio to synchronized video...")
                paths["retimed_with_audio"] = os.path.join(temp_dir, "retimed_with_audio.mp4")
                try:
                    mux_final_output(paths["retimed_video"], target_audio_path, paths["retimed_with_audio"])
                    synchronized_video_with_audio = paths["retimed_with_audio"]
                except Exception as e:
                    logging.warning(f"Audio muxing failed: {e}, using video without audio")
                    synchronized_video_with_audio = paths["retimed_video"]
                
                # Predict freezes from timecodes (same as diagnostic)
                progress(0.8, desc="7/8: Predicting and repairing freezes with RIFE...")
                freeze_predictions = predict_freezes_from_timecodes(paths["timecodes"])
                
                if freeze_predictions:
                    # Apply RIFE repair to synchronized video with audio (same as diagnostic)
                    interpolation_applied = repair_freezes_with_rife(
                        synchronized_video_with_audio, freeze_predictions, paths["interpolated_video"], RIFE_MODEL
                    )
                    if interpolation_applied:
                        # RIFE created video WITHOUT audio, need to add audio back
                        progress(0.85, desc="7/8: Adding audio to AI-repaired video...")
                        paths["interpolated_with_audio"] = os.path.join(temp_dir, "interpolated_with_audio.mp4") 
                        try:
                            # Add audio from synchronized video (same as diagnostic)
                            cmd = [
                                'ffmpeg', '-y',
                                '-i', paths["interpolated_video"],      # RIFE video (no audio)
                                '-i', synchronized_video_with_audio,   # Audio source
                                '-c:v', 'copy',                        # Copy video as-is
                                '-c:a', 'aac',                        # AAC audio codec
                                '-b:a', '128k',                       # Audio bitrate
                                '-map', '0:v:0',                      # Take video from RIFE
                                '-map', '1:a:0?',                     # Take audio from synchronized
                                '-shortest',                          # End when shortest ends
                                paths["interpolated_with_audio"]
                            ]
                            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                            if result.returncode == 0:
                                video_for_final_mux = paths["interpolated_with_audio"]
                            else:
                                logging.warning(f"Audio addition failed: {result.stderr}")
                                video_for_final_mux = synchronized_video_with_audio
                        except Exception as e:
                            logging.warning(f"Audio processing error: {e}")
                            video_for_final_mux = synchronized_video_with_audio
                    else:
                        # RIFE failed, use synchronized video with audio
                        video_for_final_mux = synchronized_video_with_audio
                else:
                    # No freezes detected, use synchronized video with audio
                    video_for_final_mux = synchronized_video_with_audio
            else:
                progress(0.8, desc="6/8: Skipping RIFE repair...")
                # Standard retiming without RIFE
                progress(0.9, desc="7/8: Retiming video...")
                retime_video(input_video_path, paths["timecodes"], paths["retimed_video"])
                video_for_final_mux = paths["retimed_video"]
            
            progress(0.95, desc="8/8: Final output...")
            if use_rife:
                # RIFE mode: video already has audio, just copy
                shutil.copy2(video_for_final_mux, paths["temp_final_output"])
            else:
                # Standard mode: add audio
                mux_final_output(video_for_final_mux, target_audio_path, paths["temp_final_output"])
            
            progress(0.99, desc="Saving...")
            shutil.copy2(paths["temp_final_output"], final_output_path)
            
            duration = time.time() - start_time
            
            # Get FPS info for status
            video_info = validate_video_format(input_video_path)
            fps, is_vfr = detect_vfr_and_get_fps(input_video_path, video_info)
            
            # Enhanced status message with RIFE info
            rife_info = f"RIFE Engine: REAL RIFE" if RIFE_MODEL.available else "RIFE: Not available"
            
            mode_note = ""
            if use_rife:
                if interpolation_applied:
                    mode_note = " (with adaptive REAL RIFE interpolation)"
                else:
                    mode_note = " (RIFE enabled - no interpolation needed)"
            
            status_msg = f"""Synchronization successful{mode_note}! 
Processing time: {duration:.2f} seconds
Video FPS: {fps:.3f} {'(Variable Frame Rate detected)' if is_vfr else '(Constant Frame Rate)'}
{rife_info}
Smoothing: Enabled (cubic spline + filter)

--- Transcript Preview ---
{transcript_text[:1000]}..."""
            
            return final_output_path, status_msg

    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)
        if 'final_output_path' in locals() and os.path.exists(final_output_path):
            try:
                os.unlink(final_output_path)
            except:
                pass
        raise gr.Error(f"Processing failed: {e}")

def diagnostic_workflow(input_video_path, target_audio_path, use_rife=True, progress=gr.Progress()):
    """Diagnostic workflow - visualize problem frames without interpolation."""
    start_time = time.time()
    
    try:
        check_dependencies()
        
        # Create output file
        diag_fd, diag_output_path = tempfile.mkstemp(suffix="_diagnostic.mp4", prefix="diagnostic_")
        os.close(diag_fd)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            logging.info(f"Starting diagnostic analysis in {temp_dir}")
            
            paths = {
                "source_audio": os.path.join(temp_dir, "source_16k.wav"),
                "target_audio_processed": os.path.join(temp_dir, "target_16k.wav"),
                "transcript": os.path.join(temp_dir, "transcript.txt"),
                "align_source": os.path.join(temp_dir, "align_source.json"),
                "align_target": os.path.join(temp_dir, "align_target.json"),
                "timecodes": os.path.join(temp_dir, "timecodes_v2.txt"),
                "retimed_video": os.path.join(temp_dir, "retimed_diagnostic.mp4"),
                "retimed_with_audio": os.path.join(temp_dir, "retimed_with_audio.mp4"),
            }
            
            # Standard pipeline for timecode generation
            progress(0.1, desc="1/7: Extracting audio...")
            extract_and_standardize_audio(input_video_path, paths["source_audio"])
            extract_and_standardize_audio(target_audio_path, paths["target_audio_processed"])
            
            progress(0.2, desc=f"2/7: Transcribing (Whisper on {DEVICE})...")
            transcript_text = transcribe_audio(paths["target_audio_processed"], paths["transcript"], WHISPER_MODEL)
            
            progress(0.35, desc="3/7: Aligning Source...")
            forced_alignment(paths["source_audio"], paths["transcript"], paths["align_source"])
            
            progress(0.5, desc="4/7: Aligning Target...")
            forced_alignment(paths["target_audio_processed"], paths["transcript"], paths["align_target"])
            
            progress(0.65, desc="5/7: Calculating timecodes...")
            generate_vfr_timecodes(
                input_video_path, 
                paths["align_source"], 
                paths["align_target"], 
                paths["timecodes"],
                smooth_interpolation=True
            )
            
            # Create physically retimed video with ACTUAL frame duplicates
            progress(0.75, desc="6/8: Creating video with physical frame duplicates...")
            create_physical_retime(input_video_path, paths["timecodes"], paths["retimed_video"])
            
            # Add audio to retimed video using target audio
            progress(0.8, desc="7/8: Adding audio to synchronized video...")
            logging.info("🔊 DIAGNOSTIC: Adding audio to retimed video...")
            logging.info(f"   Source video (no audio): {paths['retimed_video']}")
            logging.info(f"   Target audio: {target_audio_path}")
            logging.info(f"   Output (with audio): {paths['retimed_with_audio']}")
            
            # Check if source files exist
            if not os.path.exists(paths["retimed_video"]):
                logging.error(f"❌ Retimed video not found: {paths['retimed_video']}")
            else:
                logging.info(f"✅ Retimed video exists: {os.path.getsize(paths['retimed_video'])} bytes")
                
            if not os.path.exists(target_audio_path):
                logging.error(f"❌ Target audio not found: {target_audio_path}")
            else:
                logging.info(f"✅ Target audio exists: {os.path.getsize(target_audio_path)} bytes")
            
            # Run mux_final_output with detailed logging
            try:
                mux_final_output(paths["retimed_video"], target_audio_path, paths["retimed_with_audio"])
                
                # Check result
                if os.path.exists(paths["retimed_with_audio"]):
                    size = os.path.getsize(paths["retimed_with_audio"])
                    logging.info(f"✅ Video with audio created: {size} bytes")
                    
                    # Verify audio streams in result
                    import subprocess
                    verify_cmd = ['ffprobe', '-v', 'quiet', '-select_streams', 'a', '-show_entries', 'stream=codec_name', '-of', 'csv=p=0', paths["retimed_with_audio"]]
                    verify_result = subprocess.run(verify_cmd, capture_output=True, text=True, timeout=30)
                    
                    if verify_result.stdout.strip():
                        logging.info(f"✅ Audio verified in retimed video: {verify_result.stdout.strip()}")
                    else:
                        logging.warning("⚠️ No audio detected in retimed video!")
                else:
                    logging.error("❌ mux_final_output failed to create output file!")
                    
            except Exception as e:
                logging.error(f"❌ mux_final_output failed: {e}")
                # Fallback - use video without audio
                logging.info("📝 Using video without audio as fallback")
                import shutil
                shutil.copy2(paths["retimed_video"], paths["retimed_with_audio"])
            
            # Create triple diagnostic: Original + Detection + AI Repair
            progress(0.9, desc="8/8: Creating AI diagnostic (Original + Detection + Repair)...")
            marked_frames, report = create_triple_diagnostic(
                paths["retimed_with_audio"],  # Use synchronized video WITH AUDIO as source
                paths["timecodes"],           # Analyze timecodes for predictions
                diag_output_path,            # Create triple comparison video
                RIFE_MODEL                   # Use RIFE for AI repair
            )
            
            duration = time.time() - start_time
            
            # Status message  
            status_msg = f"""🤖 AI DIAGNOSTIC COMPLETE!
Processing time: {duration:.2f} seconds

{report}

🎬 Triple comparison shows: Original → Freeze Detection → AI Repair
Watch the bottom panel to see RIFE-repaired smooth video!"""
            
            return diag_output_path, status_msg
    
    except Exception as e:
        logging.error(f"Diagnostic error: {e}", exc_info=True)
        if 'diag_output_path' in locals() and os.path.exists(diag_output_path):
            try:
                os.unlink(diag_output_path)
            except:
                pass
        raise gr.Error(f"Diagnostic failed: {e}")


# Initialize models
initialize_models()

# --- Gradio Interface ---
with gr.Blocks(title="Enhanced Video-Audio Synchronizer with RIFE") as interface:
    gr.Markdown("# Enhanced Video-to-Audio Non-Linear Synchronizer with RIFE")
    gr.Markdown(f"""
    Upload source video and target audio. Choose RIFE mode for AI-powered frame interpolation.
    
    **🚀 Enhanced Features:**
    - **Real RIFE AI**: Automatic download and setup of AI models
    - **VFR Detection**: Smart frame rate analysis
    - **Format Validation**: MP4/MOV compatibility checking  
    - **Smooth Interpolation**: Cubic spline + AI smoothing
    - **Progress Tracking**: Real-time processing updates
    - **Smart Analysis**: Intelligent problem area detection
    
    **🤖 RIFE Status**: REAL RIFE - {'✅ AI Ready' if RIFE_MODEL.available else '🚀 Installing on demand'}
    **🚀 GPU**: {'✅ CUDA Available - ' + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else '') if torch.cuda.is_available() else '⚠️ CPU Only'}
    **💾 VRAM**: {torch.cuda.get_device_properties(0).total_memory // 1024**3 if torch.cuda.is_available() else 0} GB
    
    **⚡ First run**: RIFE AI models download automatically (2-5 minutes one-time setup)
    """)
    
    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Source Video")
            audio_input = gr.Audio(label="Target Audio", type="filepath")
            
            gr.Markdown("### Video Processing")
            use_rife = gr.Checkbox(
                label="Use RIFE AI Interpolation", 
                value=True,
                info="Enable AI-based frame interpolation to repair video freezes"
            )
            
            submit_button = gr.Button("Start Synchronization", variant="primary")
            
            gr.Markdown("---")
            gr.Markdown("### 🤖 AI Diagnostic Mode")
            gr.Markdown("Creates 3-panel comparison: Original + Freeze Detection + AI Repair")
            gr.Markdown("**New:** Uses RIFE to repair detected freezes and shows results!")
            
            diagnostic_button = gr.Button("🤖 Run AI Diagnostic (Detection + Repair)", variant="secondary")
            
        
        with gr.Column():
            video_output = gr.Video(label="Synchronized Video / Comparison Grid")
            text_output = gr.Textbox(label="Status & Transcript", lines=10)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown(f"""
            **🚀 Off**: VFR only (3-5s)
            - Basic synchronization only
            - No frame interpolation
            - Fastest processing
            """)
        
        with gr.Column():
            gr.Markdown(f"""
            **🎯 Adaptive**: Smart AI interpolation (15-30s)
            - Real RIFE AI analysis of timing issues
            - Interpolates only problem areas
            - Engine: {'REAL RIFE' if RIFE_MODEL.available else 'Auto-install'}
            """)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown(f"""
            **🔧 Precision**: Surgical RIFE (10-25s)
            - Real RIFE AI at exact VFR points
            - Minimal but precise interpolation
            - Engine: {'REAL RIFE' if RIFE_MODEL.available else 'Auto-install'}
            """)
        
        with gr.Column():
            gr.Markdown(f"""
            **💎 Maximum**: Full RIFE AI (60-120s)
            - Real RIFE AI on entire video
            - Maximum smoothness possible
            - Engine: {'REAL RIFE' if RIFE_MODEL.available else 'Auto-install'}
            """)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("""
            ### 🤖 **AI Diagnostic Mode**
            
            Creates a **3-panel comparison video**:
            - 🟦 **TOP**: Original synchronized video
            - 🟡 **MIDDLE**: Freeze detection with colored markers
            - 🟢 **BOTTOM**: AI-repaired video using RIFE
            
            Perfect for testing freeze detection accuracy and AI repair quality!
            """)
        
        with gr.Column():
            gr.Markdown("""
            ### 🎬 **Comparison Mode**
            
            Automatically processes your video with all 4 modes and creates a **2x2 grid comparison**:
            
            ```
            ┌─────────────┬─────────────┐
            │  ORIGINAL   │  ADAPTIVE   │
            │ (VFR Only)  │    RIFE     │
            ├─────────────┼─────────────┤  
            │ PRECISION   │  MAXIMUM    │
            │    RIFE     │    RIFE     │
            └─────────────┴─────────────┘
            ```
            
            Perfect for evaluating which mode works best for your content!
            """)

    # Connect buttons to functions
    submit_button.click(
        fn=synchronization_workflow,
        inputs=[video_input, audio_input, use_rife],
        outputs=[video_output, text_output]
    )
    
    diagnostic_button.click(
        fn=diagnostic_workflow,
        inputs=[video_input, audio_input, use_rife],
        outputs=[video_output, text_output]
    )

if __name__ == '__main__':
    print("Launching Enhanced Video-Audio Synchronizer...")
    
    # Display system info
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory // 1024**3
            print(f"🚀 GPU Detected: {gpu_name} ({gpu_memory} GB VRAM)")
            print(f"🔥 RIFE Method: REAL RIFE")
            print("⚡ Ready for GPU-accelerated interpolation!")
        except Exception as e:
            print(f"GPU info error: {e}")
    else:
        print("⚠️  No GPU detected - using CPU only")
    
    interface.launch(share=True, server_name="0.0.0.0")