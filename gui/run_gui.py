"""
Run the Streamlit GUI application.

This script serves as the entry point for the Streamlit web application.
It can be run from the command line using: streamlit run run_gui.py
"""

import os
import sys
import subprocess
import time
import socket
import logging

# Setup optimized logging (only enable debug if needed)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Enable debug logging only if environment variable is set
debug_mode = os.getenv('AI_PPO_DEBUG', 'false').lower() == 'true'
if debug_mode:
    logging.basicConfig(level=logging.DEBUG)
    print("🔍 Debug mode enabled")
else:
    print("⚡ OPTIMIZED LOGGING ENABLED")

def is_port_in_use(port):
    """Check if a port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            return False
        except OSError:
            return True

def main():
    """Main function to run the Streamlit app."""
    
    # Add src to path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(current_dir, '..', 'src')
    sys.path.insert(0, src_path)
    
    # Set environment variables
    os.environ['PYTHONPATH'] = src_path
    
    # Path to the main app
    app_path = os.path.join(current_dir, 'app.py')
    
    # Find an available port
    ports_to_try = [8501, 8502, 8503, 8504, 8505]
    available_port = None
    
    for port in ports_to_try:
        if not is_port_in_use(port):
            available_port = port
            break
    
    if available_port is None:
        print("❌ No available ports found. Please close other Streamlit instances.")
        sys.exit(1)
    
    print(f"🚀 Starting Streamlit on port {available_port}...")
    print(f"🌐 URL: http://localhost:{available_port}")
    
    # Change to the GUI directory
    os.chdir(current_dir)
    
    # Use os.system to run Streamlit (this avoids the 95% hang issue)
    command = f'streamlit run app.py --server.port {available_port} --server.address localhost --browser.gatherUsageStats false'
    os.system(command)

if __name__ == "__main__":
    main()