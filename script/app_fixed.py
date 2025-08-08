import subprocess
import os
import json
import numpy as np
from scipy.interpolate import interp1d
import cv2
import whisper
import logging
import shutil
import gradio as gr
import tempfile
import time
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

# --- Global Setup: Load Whisper Model Once ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WHISPER_MODEL = None
try:
    logging.info(f"Loading Whisper model ('base') to {DEVICE}... (This speeds up subsequent requests)")
    # Whisper automatically uses the GPU if available on RunPod.
    WHISPER_MODEL = whisper.load_model("base", device=DEVICE) 
except Exception as e:
    logging.error(f"Failed to load Whisper model: {e}")

# --- Core Utility Functions ---

def check_dependencies():
    """Check if required external tools are available."""
    missing = []
    if not shutil.which("mp4fpsmod"): missing.append("mp4fpsmod")
    if not shutil.which("ffmpeg"): missing.append("ffmpeg")
    if not shutil.which("espeak-ng") and not shutil.which("espeak"):
         missing.append("espeak/espeak-ng")
    if missing:
        raise EnvironmentError(f"Missing dependencies: {', '.join(missing)}.")

def run_command(command):
    """Helper function to run shell commands."""
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        # Raise an error that can be caught and displayed in the UI
        raise RuntimeError(f"Command failed: {result.stderr}")
    return result.stdout

# --- Synchronization Steps ---

def extract_and_standardize_audio(input_path, output_audio_path):
    # Extract audio to 16kHz Mono PCM for ASR/Alignment
    command = [
        "ffmpeg", "-y", "-i", input_path, "-vn", "-acodec", "pcm_s16le", 
        "-ar", "16000", "-ac", "1", output_audio_path
    ]
    run_command(command)

def transcribe_audio(audio_path, output_transcript_path):
    if WHISPER_MODEL is None: raise RuntimeError("Whisper model not loaded.")
    result = WHISPER_MODEL.transcribe(audio_path)
    text = result["text"]
    # Format text for Aeneas (breaking lines improves alignment)
    formatted_text = text.strip().replace('. ', '.\n').replace('? ', '?\n')
    with open(output_transcript_path, 'w') as f:
        f.write(formatted_text) 
    return formatted_text

def forced_alignment(audio_path, transcript_path, output_alignment_path):
    # Use Aeneas to find word timestamps
    # We use python3 explicitly as it's standard in modern Linux environments
    command = [
        "python3", "-m", "aeneas.tools.execute_task",
        audio_path, transcript_path,
        "task_language=eng|os_task_file_format=json|is_text_type=plain",
        output_alignment_path
    ]
    run_command(command)

def generate_vfr_timecodes(video_path, align_source_path, align_target_path, output_timecode_path):
    # The "Warp Engine": Calculate time mapping
    with open(align_source_path) as f: align_source = json.load(f)
    with open(align_target_path) as f: align_target = json.load(f)

    T_source, T_target = [0.0], [0.0]
    # Use the minimum length in case ASR resulted in slight mismatches
    min_len = min(len(align_source['fragments']), len(align_target['fragments']))
    
    for i in range(min_len):
        T_source.append(float(align_source['fragments'][i]['end']))
        T_target.append(float(align_target['fragments'][i]['end']))

    # Create interpolation function (Source time -> Target time)
    alignment_func = interp1d(T_source, T_target, bounds_error=False, fill_value="extrapolate")

    # Get video details
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Map every frame to the new timeline
    original_timestamps = np.arange(total_frames) / video_fps
    new_timestamps_ms = (alignment_func(original_timestamps) * 1000).round().astype(int)

    # Sanitize timestamps (ensure start at 0)
    new_timestamps_ms = np.maximum(0, new_timestamps_ms)
    if len(new_timestamps_ms) > 0 and new_timestamps_ms[0] != 0:
        new_timestamps_ms -= new_timestamps_ms[0]

    # Write V2 timecode file (must be strictly monotonically increasing)
    with open(output_timecode_path, 'w') as f:
        f.write("# timecode format v2\n")
        prev_timestamp = -1
        for timestamp in new_timestamps_ms:
            # CRITICAL: If timestamps collide due to rounding/speed changes, force an increment
            if timestamp <= prev_timestamp:
                timestamp = prev_timestamp + 1
            f.write(f"{timestamp}\n")
            prev_timestamp = timestamp

def retime_video(input_video_path, timecode_path, output_retimed_video_path):
    # Apply timestamps losslessly (mp4fpsmod)
    binary = shutil.which("mp4fpsmod")
    command = [binary, "-t", timecode_path, "-o", output_retimed_video_path, input_video_path]
    run_command(command)

def mux_final_output(retimed_video_path, target_audio_path, final_output_path):
    # Combine retimed video with target audio (FFmpeg)
    command = [
        "ffmpeg", "-y", "-i", retimed_video_path, "-i", target_audio_path,
        "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0",
        "-shortest", final_output_path
    ]
    run_command(command)


# --- Gradio Workflow Wrapper ---

def synchronization_workflow(input_video_path, target_audio_path, progress=gr.Progress()):
    """The main function that the Web UI calls."""
    start_time = time.time()
    try:
        check_dependencies()
        
        # Create a persistent temporary file for the final output
        final_output_fd, final_output_path = tempfile.mkstemp(suffix=".mp4", prefix="synchronized_")
        os.close(final_output_fd)  # Close the file descriptor, we just need the path
        
        # Use a temporary directory for processing (automatically cleaned up)
        with tempfile.TemporaryDirectory() as temp_dir:
            logging.info(f"Starting processing in {temp_dir}")

            # Define paths
            paths = {
                "source_audio": os.path.join(temp_dir, "source_16k.wav"),
                "target_audio_processed": os.path.join(temp_dir, "target_16k.wav"),
                "transcript": os.path.join(temp_dir, "transcript.txt"),
                "align_source": os.path.join(temp_dir, "align_source.json"),
                "align_target": os.path.join(temp_dir, "align_target.json"),
                "timecodes": os.path.join(temp_dir, "timecodes_v2.txt"),
                "retimed_video": os.path.join(temp_dir, "retimed.mp4"),
                "temp_final_output": os.path.join(temp_dir, "synchronized_output.mp4")
            }

            # --- Pipeline Execution ---
            progress(0.1, desc="1/7: Extracting audio...")
            extract_and_standardize_audio(input_video_path, paths["source_audio"])
            extract_and_standardize_audio(target_audio_path, paths["target_audio_processed"])
            
            progress(0.2, desc=f"2/7: Transcribing target audio (Whisper on {DEVICE})...")
            transcript_text = transcribe_audio(paths["target_audio_processed"], paths["transcript"])
            
            progress(0.4, desc="3/7: Aligning Source (Aeneas)...")
            forced_alignment(paths["source_audio"], paths["transcript"], paths["align_source"])

            progress(0.6, desc="4/7: Aligning Target (Aeneas)...")
            forced_alignment(paths["target_audio_processed"], paths["transcript"], paths["align_target"])
            
            progress(0.8, desc="5/7: Calculating time warp map (VFR)...")
            generate_vfr_timecodes(input_video_path, paths["align_source"], paths["align_target"], paths["timecodes"])
            
            progress(0.9, desc="6/7: Retiming video losslessly (mp4fpsmod)...")
            retime_video(input_video_path, paths["timecodes"], paths["retimed_video"])
            
            progress(0.95, desc="7/7: Muxing final output (FFmpeg)...")
            # Use the original high-quality target audio for the final mux
            mux_final_output(paths["retimed_video"], target_audio_path, paths["temp_final_output"])
            
            # Copy the final output to a persistent location
            progress(0.99, desc="Saving final output...")
            shutil.copy2(paths["temp_final_output"], final_output_path)
            
            duration = time.time() - start_time
            status_msg = f"Synchronization successful! Processed in {duration:.2f} seconds.\n\n--- Transcript Preview ---\n{transcript_text[:1000]}..."
            return final_output_path, status_msg

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        # Clean up the persistent file if there was an error
        if 'final_output_path' in locals() and os.path.exists(final_output_path):
            try:
                os.unlink(final_output_path)
            except:
                pass
        # Display the error in the UI
        raise gr.Error(f"Processing failed: {e}")

# --- Gradio UI Definition ---

with gr.Blocks(title="Video-Audio Synchronisator") as interface:
    gr.Markdown("# Video-to-Audio Non-Linear Synchronizer")
    gr.Markdown("Upload the source video (e.g., Veo 3) and the target guided audio. The video's timing will be losslessly warped (stretched/condensed) to match the target audio.")
    
    with gr.Row():
        with gr.Column():
            # Use gr.Video for easy upload and preview
            video_input = gr.Video(label="Source Video (Input)")
            # Use gr.Audio with type="filepath" to ensure the backend gets the file path string
            audio_input = gr.Audio(label="Target Guided Audio (Input)", type="filepath")
            submit_button = gr.Button("Start Synchronization", variant="primary")
        
        with gr.Column():
            video_output = gr.Video(label="Synchronized Video (Output)")
            text_output = gr.Textbox(label="Processing Status and Transcript", lines=10)

    # Link the button to the workflow function
    submit_button.click(
        fn=synchronization_workflow,
        inputs=[video_input, audio_input],
        outputs=[video_output, text_output]
    )

# Launch the interface
if __name__ == '__main__':
    print("Launching Gradio Web Interface...")
    # share=True generates a temporary public link (lasts 72 hours) for easy access.
    # server_name="0.0.0.0" is crucial for making the app accessible externally on RunPod/Docker.
    interface.launch(share=True, server_name="0.0.0.0")