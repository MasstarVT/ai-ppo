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

# Setup optimized logging (only enable debug if needed)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from debug_config import setup_debug_logging

# Enable debug logging only if environment variable is set
debug_mode = os.getenv('AI_PPO_DEBUG', 'false').lower() == 'true'
setup_debug_logging(enable_debug=debug_mode)

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
    
    # Add project root and src to path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.abspath(os.path.join(current_dir, '..'))
    src_path = os.path.join(root_path, 'src')
    for p in (root_path, src_path):
        if p not in sys.path:
            sys.path.insert(0, p)

    # Set environment variables for child process
    existing_pyspath = os.environ.get('PYTHONPATH', '')
    parts = [root_path, src_path]
    if existing_pyspath:
        parts.append(existing_pyspath)
    os.environ['PYTHONPATH'] = os.pathsep.join(parts)
    
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
    
    # Run Streamlit
    try:
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', app_path,
            '--server.port', str(available_port),
            '--server.address', 'localhost',
            '--browser.gatherUsageStats', 'false'
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Streamlit: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 Streamlit app stopped by user")
        sys.exit(0)

if __name__ == "__main__":
    main()