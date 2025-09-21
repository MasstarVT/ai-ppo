#!/usr/bin/env python3
"""
Simple test to run training for 20 seconds and see what happens
"""

import subprocess
import sys
import os
import time

def test_training():
    """Test training for 20 seconds to see if it crashes."""
    print("🧪 Testing training crash issue...")
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Command that mimics GUI training start
    cmd = [
        sys.executable, 
        os.path.join(project_root, "train_enhanced.py"),
        "--mode", "new",
        "--timesteps", "1000",  # Small number for quick test
        "--network-size", "small"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print("Starting training process...")
    
    try:
        # Start process similar to GUI
        process = subprocess.Popen(
            cmd,
            cwd=project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Monitor for 20 seconds
        start_time = time.time()
        while time.time() - start_time < 20:
            # Check if process is still running
            return_code = process.poll()
            if return_code is not None:
                # Process has terminated
                stdout, stderr = process.communicate()
                print(f"❌ Process terminated after {time.time() - start_time:.1f} seconds")
                print(f"Return code: {return_code}")
                if stdout:
                    print(f"STDOUT:\n{stdout}")
                if stderr:
                    print(f"STDERR:\n{stderr}")
                return False
            
            # Still running
            print(f"✅ Training running for {time.time() - start_time:.1f} seconds...")
            time.sleep(2)
        
        # Kill process after test
        process.terminate()
        process.wait()
        print("✅ Training ran successfully for 20 seconds without crashing")
        return True
        
    except Exception as e:
        print(f"❌ Error running training: {e}")
        return False

if __name__ == "__main__":
    test_training()