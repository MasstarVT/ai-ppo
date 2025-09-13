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

# Import and setup debug logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from debug_config import setup_debug_logging

print("🐛 Setting up debug logging...")
setup_debug_logging()
print("✅ Debug logging initialized")

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