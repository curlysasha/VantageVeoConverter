"""
Utility functions for managing binary dependencies (ffmpeg, mp4fpsmod, etc.)
"""
import os
import shutil
import logging

def get_binary_path(binary_name):
    """
    Get path to a binary, checking multiple locations:
    1. Local bin/ directory relative to this script
    2. bin/ directory relative to main app file
    3. System PATH
    
    Args:
        binary_name: Name of the binary (e.g., 'ffmpeg', 'ffprobe', 'mp4fpsmod')
    
    Returns:
        Path to the binary or None if not found
    """
    # Get current script directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try multiple possible project root directories
    possible_roots = [
        os.path.dirname(current_dir),  # ../bin (relative to src/)
        os.getcwd(),                   # ./bin (current working directory)  
        os.path.dirname(os.getcwd()),  # ../bin (if running from subdirectory)
    ]
    
    # Try to find main app file and use its directory
    try:
        import sys
        for path in sys.path:
            app_path = os.path.join(path, "app_rife_compact.py")
            if os.path.exists(app_path):
                possible_roots.insert(0, path)
                break
    except:
        pass
    
    # Check each possible location
    for project_dir in possible_roots:
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
                    continue
            
            logging.info(f"Using local {binary_name}: {local_binary}")
            return local_binary
    
    # Try environment variable override
    env_var = f"{binary_name.upper()}_PATH"
    env_path = os.environ.get(env_var)
    if env_path and os.path.exists(env_path):
        logging.info(f"Using {binary_name} from {env_var}: {env_path}")
        return env_path
    
    # Fallback to system PATH
    system_binary = shutil.which(binary_name)
    if system_binary:
        logging.info(f"Using system {binary_name}: {system_binary}")
        return system_binary
    
    # Show all locations we tried
    logging.warning(f"{binary_name} not found! Tried locations:")
    for project_dir in possible_roots:
        test_path = os.path.join(project_dir, "bin", binary_name)
        if os.name == 'nt' and not test_path.endswith('.exe'):
            test_path += '.exe'
        logging.warning(f"  - {test_path}")
    logging.warning(f"  - System PATH: {shutil.which(binary_name)}")
    logging.warning(f"  - Environment var {env_var}: {env_path}")
    
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