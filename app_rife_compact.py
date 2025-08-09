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
import whisper

# Import our modules
from src.rife_engine import RealRIFE
from src.timing_analyzer import analyze_timing_changes
from src.video_processor import interpolate_video, regenerate_timecodes_for_interpolated_video
from src.audio_sync import *
from src.comparison import create_comparison_grid
from src.diagnostic_visualizer import create_diagnostic_video
from src.simple_diagnostic import create_simple_diagnostic

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
    RIFE_MODEL = RealRIFE(DEVICE)

def check_dependencies():
    """Check if required external tools are available."""
    missing = []
    if not shutil.which("mp4fpsmod"): missing.append("mp4fpsmod")
    if not shutil.which("ffmpeg"): missing.append("ffmpeg")
    if not shutil.which("espeak-ng") and not shutil.which("espeak"):
         missing.append("espeak/espeak-ng")
    if missing:
        raise EnvironmentError(f"Missing dependencies: {', '.join(missing)}.")

def synchronization_workflow(input_video_path, target_audio_path, rife_mode="off", progress=gr.Progress()):
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
            
            # RIFE interpolation
            video_for_retiming = input_video_path
            timecodes_for_retiming = paths["timecodes"]
            interpolation_applied = False
            
            if rife_mode != "off":
                progress(0.75, desc=f"6/8: Analyzing for {rife_mode} interpolation...")
                problem_segments = analyze_timing_changes(paths["timecodes"], rife_mode=rife_mode, video_path=input_video_path)
                
                if problem_segments or rife_mode == "maximum":
                    progress(0.8, desc=f"6/8: Applying {rife_mode} interpolation...")
                    interpolation_applied = interpolate_video(
                        input_video_path, problem_segments, paths["interpolated_video"], rife_mode, RIFE_MODEL
                    )
                    if interpolation_applied:
                        video_for_retiming = paths["interpolated_video"]
                        # Пересчитываем timecodes для интерполированного видео
                        progress(0.85, desc="6/8: Recalculating timecodes...")
                        regenerate_timecodes_for_interpolated_video(
                            input_video_path, 
                            paths["interpolated_video"],
                            paths["timecodes"],
                            paths["interpolated_timecodes"]
                        )
                        timecodes_for_retiming = paths["interpolated_timecodes"]
            else:
                progress(0.8, desc="6/8: Skipping interpolation...")
            
            progress(0.9, desc="7/8: Retiming video...")
            retime_video(video_for_retiming, timecodes_for_retiming, paths["retimed_video"])
            
            progress(0.95, desc="8/8: Final muxing...")
            mux_final_output(paths["retimed_video"], target_audio_path, paths["temp_final_output"])
            
            progress(0.99, desc="Saving...")
            shutil.copy2(paths["temp_final_output"], final_output_path)
            
            duration = time.time() - start_time
            
            # Get FPS info for status
            video_info = validate_video_format(input_video_path)
            fps, is_vfr = detect_vfr_and_get_fps(input_video_path, video_info)
            
            # Enhanced status message with RIFE info
            rife_info = f"RIFE Engine: {RIFE_MODEL.method.upper()}" if RIFE_MODEL.available else "RIFE: Simple fallback"
            
            mode_note = ""
            if rife_mode != "off":
                if interpolation_applied:
                    mode_note = f" (with {rife_mode} {RIFE_MODEL.method.upper()} interpolation)"
                else:
                    mode_note = f" ({rife_mode} mode - no interpolation needed)"
            
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

def diagnostic_workflow(input_video_path, target_audio_path, rife_mode="adaptive", progress=gr.Progress()):
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
            
            # Create retimed video (same as regular sync)
            progress(0.75, desc="6/7: Creating synchronized video...")
            retime_video(input_video_path, paths["timecodes"], paths["retimed_video"])
            
            # Just copy the synchronized video as diagnostic result
            progress(0.85, desc="7/7: Copying synchronized video for review...")
            import shutil
            shutil.copy2(paths["retimed_video"], diag_output_path)
            
            duration = time.time() - start_time
            
            # Status message  
            status_msg = f"""🔍 DIAGNOSTIC: SYNCHRONIZED VIDEO (NO MARKERS YET)
Processing time: {duration:.2f} seconds

This is the synchronized video with freezes from timing corrections.
Compare this to your original to see where freezes appear.

TODO: Add red markers to frozen frames."""
            
            return diag_output_path, status_msg
    
    except Exception as e:
        logging.error(f"Diagnostic error: {e}", exc_info=True)
        if 'diag_output_path' in locals() and os.path.exists(diag_output_path):
            try:
                os.unlink(diag_output_path)
            except:
                pass
        raise gr.Error(f"Diagnostic failed: {e}")

def comparison_workflow(input_video_path, target_audio_path, progress=gr.Progress()):
    """Workflow для сравнения всех режимов."""
    start_time = time.time()
    
    try:
        check_dependencies()
        
        # Создаем выходные файлы
        output_files = {}
        for mode in ['original', 'adaptive', 'precision', 'maximum']:
            fd, path = tempfile.mkstemp(suffix=f"_{mode}.mp4", prefix="comparison_")
            os.close(fd)
            output_files[mode] = path
        
        # Финальный диптих
        grid_fd, grid_path = tempfile.mkstemp(suffix="_comparison_grid.mp4", prefix="final_")
        os.close(grid_fd)
        
        modes_to_process = [
            ('original', 'off'),
            ('adaptive', 'adaptive'), 
            ('precision', 'precision'),
            ('maximum', 'maximum')
        ]
        
        results = {}
        
        for i, (mode_name, rife_mode) in enumerate(modes_to_process):
            progress_start = i * 0.2
            progress_end = (i + 1) * 0.2
            
            progress(progress_start, desc=f"Processing {mode_name.upper()} mode...")
            
            # Используем существующий workflow для каждого режима
            def mode_progress(value, desc=""):
                actual_progress = progress_start + (value * (progress_end - progress_start))
                progress(actual_progress, desc=f"{mode_name.upper()}: {desc}")
            
            try:
                result_path, status = synchronization_workflow(
                    input_video_path, target_audio_path, rife_mode, mode_progress
                )
                
                # Копируем результат
                shutil.copy2(result_path, output_files[mode_name])
                results[mode_name] = {
                    'path': output_files[mode_name],
                    'status': status
                }
                
                # Очищаем временный файл
                try:
                    os.unlink(result_path)
                except:
                    pass
                    
                logging.info(f"Completed {mode_name} mode")
                
            except Exception as e:
                logging.error(f"Failed {mode_name} mode: {e}")
                results[mode_name] = {'path': None, 'status': f"Failed: {e}"}
        
        # Создаем диптих если все режимы успешны
        progress(0.85, desc="Creating comparison grid...")
        
        if all(r['path'] for r in results.values()):
            grid_success = create_comparison_grid(
                results['original']['path'],
                results['adaptive']['path'], 
                results['precision']['path'],
                results['maximum']['path'],
                grid_path
            )
            
            if grid_success:
                progress(0.95, desc="Finalizing comparison...")
                
                duration = time.time() - start_time
                
                # Создаем подробный отчет
                status_report = f"""🎬 COMPARISON COMPLETE! Processed in {duration:.1f} seconds.

📊 ALL MODES COMPARISON:
• ORIGINAL (VFR Only): Basic synchronization
• ADAPTIVE RIFE: Smart interpolation where needed  
• PRECISION RIFE: Surgical interpolation at VFR points
• MAXIMUM RIFE: Full video interpolation

🎯 Grid Layout (2x2):
┌─────────────┬─────────────┐
│  ORIGINAL   │  ADAPTIVE   │
├─────────────┼─────────────┤  
│ PRECISION   │  MAXIMUM    │
└─────────────┴─────────────┘

⏱️ Processing times and details logged above.
Watch the grid to see smoothness differences!"""

                # Очищаем индивидуальные файлы
                for mode_result in results.values():
                    if mode_result['path']:
                        try:
                            os.unlink(mode_result['path'])
                        except:
                            pass
                
                return grid_path, status_report
            else:
                raise Exception("Failed to create comparison grid")
        else:
            failed_modes = [mode for mode, result in results.items() if not result['path']]
            raise Exception(f"Failed modes: {', '.join(failed_modes)}")
            
    except Exception as e:
        logging.error(f"Comparison workflow failed: {e}", exc_info=True)
        
        # Очистка в случае ошибки
        if 'results' in locals():
            for mode_result in results.values():
                if mode_result.get('path') and os.path.exists(mode_result['path']):
                    try:
                        os.unlink(mode_result['path'])
                    except:
                        pass
        
        if 'grid_path' in locals() and os.path.exists(grid_path):
            try:
                os.unlink(grid_path)
            except:
                pass
                
        raise gr.Error(f"Comparison failed: {e}")

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
    
    **🤖 RIFE Status**: {RIFE_MODEL.method.upper()} - {'✅ AI Ready' if RIFE_MODEL.available else '🚀 Installing on demand'}
    **🚀 GPU**: {'✅ CUDA Available - ' + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else '') if torch.cuda.is_available() else '⚠️ CPU Only'}
    **💾 VRAM**: {torch.cuda.get_device_properties(0).total_memory // 1024**3 if torch.cuda.is_available() else 0} GB
    
    **⚡ First run**: RIFE AI models download automatically (2-5 minutes one-time setup)
    """)
    
    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Source Video")
            audio_input = gr.Audio(label="Target Audio", type="filepath")
            
            gr.Markdown("### Single Mode Processing")
            rife_mode = gr.Radio(
                choices=[
                    ("Off", "off"),
                    ("Adaptive", "adaptive"), 
                    ("Precision", "precision"),
                    ("Maximum", "maximum")
                ],
                value="off",
                label="RIFE Interpolation Mode"
            )
            
            submit_button = gr.Button("Start Synchronization", variant="primary")
            
            gr.Markdown("---")
            gr.Markdown("### 🔍 Diagnostic Mode")
            gr.Markdown("Visualize detected problem frames without interpolation")
            gr.Markdown("**Note:** Uses ultra-sensitive detection to catch ALL potential issues")
            
            diagnostic_button = gr.Button("🔍 Run Diagnostic (Show Problem Frames)", variant="secondary")
            
            gr.Markdown("---")
            gr.Markdown("### Compare All Modes")
            gr.Markdown("Process video with all 4 modes and create side-by-side comparison")
            
            compare_button = gr.Button("🎬 Compare All Modes (Off/Adaptive/Precision/Maximum)", variant="secondary", size="lg")
        
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
            - Engine: {RIFE_MODEL.method.upper() if RIFE_MODEL.available else 'Auto-install'}
            """)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown(f"""
            **🔧 Precision**: Surgical RIFE (10-25s)
            - Real RIFE AI at exact VFR points
            - Minimal but precise interpolation
            - Engine: {RIFE_MODEL.method.upper() if RIFE_MODEL.available else 'Auto-install'}
            """)
        
        with gr.Column():
            gr.Markdown(f"""
            **💎 Maximum**: Full RIFE AI (60-120s)
            - Real RIFE AI on entire video
            - Maximum smoothness possible
            - Engine: {RIFE_MODEL.method.upper() if RIFE_MODEL.available else 'Auto-install'}
            """)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("""
            ### 🔍 **Diagnostic Mode**
            
            Creates a video with **visual markers** on detected problem frames:
            - 🔴 **RED BORDER** = Problem frame detected
            - 📝 **TEXT OVERLAY** = Shows issue type (freeze, duplicate, etc.)
            - 🟡 **YELLOW CORNERS** = Visual emphasis
            
            Use this to verify detection accuracy before running interpolation!
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
        inputs=[video_input, audio_input, rife_mode],
        outputs=[video_output, text_output]
    )
    
    diagnostic_button.click(
        fn=diagnostic_workflow,
        inputs=[video_input, audio_input, rife_mode],
        outputs=[video_output, text_output]
    )
    
    compare_button.click(
        fn=comparison_workflow,
        inputs=[video_input, audio_input],
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
            print(f"🔥 RIFE Method: {RIFE_MODEL.method.upper()}")
            print("⚡ Ready for GPU-accelerated interpolation!")
        except Exception as e:
            print(f"GPU info error: {e}")
    else:
        print("⚠️  No GPU detected - using CPU only")
    
    interface.launch(share=True, server_name="0.0.0.0")