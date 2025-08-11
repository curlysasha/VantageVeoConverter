"""
Utility functions for managing binary dependencies (ffmpeg, mp4fpsmod, etc.)
"""
import os
import shutil
import logging

def get_binary_path(binary_name):
    """
    Get path to a binary, checking local bin/ directory first, then system PATH.
    
    Args:
        binary_name: Name of the binary (e.g., 'ffmpeg', 'ffprobe', 'mp4fpsmod')
    
    Returns:
        Path to the binary or None if not found
    """
    # Get project root directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)
    
    # Check for local binary in project/bin directory
    local_binary = os.path.join(project_dir, "bin", binary_name)
    
    # Add .exe extension for Windows
    if os.name == 'nt':
        if not local_binary.endswith('.exe'):
            local_binary += '.exe'
    
    # Check if local binary exists and is executable
    if os.path.exists(local_binary):
        if os.name != 'nt' and not os.access(local_binary, os.X_OK):
            logging.warning(f"Found {local_binary} but it's not executable. Trying to fix...")
            try:
                os.chmod(local_binary, 0o755)
                logging.info(f"Made {local_binary} executable")
            except Exception as e:
                logging.error(f"Failed to make {local_binary} executable: {e}")
        
        logging.info(f"Using local {binary_name}: {local_binary}")
        return local_binary
    
    # Fallback to system PATH
    system_binary = shutil.which(binary_name)
    if system_binary:
        logging.info(f"Using system {binary_name}: {system_binary}")
        return system_binary
    
    logging.warning(f"{binary_name} not found in {local_binary} or system PATH")
    return None

def get_ffmpeg():
    """Get path to ffmpeg binary."""
    return get_binary_path("ffmpeg")

def get_ffprobe():
    """Get path to ffprobe binary."""
    return get_binary_path("ffprobe")

def get_mp4fpsmod():
    """Get path to mp4fpsmod binary."""
    return get_binary_path("mp4fpsmod")

def check_all_binaries():
    """
    Check all required binaries and return missing ones.
    
    Returns:
        List of missing binaries
    """
    missing = []
    
    if not get_ffmpeg():
        missing.append("ffmpeg")
    
    if not get_ffprobe():
        missing.append("ffprobe")
    
    if not get_mp4fpsmod():
        missing.append("mp4fpsmod")
    
    return missing