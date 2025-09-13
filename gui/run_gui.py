"""
Run the Streamlit GUI application.

This script serves as the entry point for the Streamlit web application.
It can be run from the command line using: streamlit run run_gui.py
"""

import os
import sys
import subprocess

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
    
    # Run Streamlit
    try:
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', app_path,
            '--server.port', '8501',
            '--server.address', 'localhost',
            '--browser.gatherUsageStats', 'false'
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nStreamlit app stopped by user")
        sys.exit(0)

if __name__ == "__main__":
    main()