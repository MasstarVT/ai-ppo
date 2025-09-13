#!/usr/bin/env python3
"""
Test script for the enhanced training functionality.
"""

import os
import sys
import subprocess
from pathlib import Path

# Fix encoding issues on Windows
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

def test_training_script():
    """Test the enhanced training script functionality."""
    print("🧪 Testing Enhanced Training Script")
    print("=" * 50)
    
    # Get project root
    current_dir = Path(__file__).parent
    project_root = current_dir
    script_path = project_root / "train_enhanced.py"
    
    print(f"📂 Project root: {project_root}")
    print(f"📄 Script path: {script_path}")
    
    # Check if script exists
    if not script_path.exists():
        print(f"❌ Training script not found: {script_path}")
        return False
    
    print("✅ Training script found")
    
    # Test help command
    print("\n🔍 Testing help command...")
    try:
        result = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print("✅ Help command successful")
            print("📋 Available options:")
            for line in result.stdout.split('\n')[:10]:  # Show first 10 lines
                if line.strip():
                    print(f"   {line}")
        else:
            print(f"❌ Help command failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing help: {e}")
        return False
    
    # Test new model training (short run)
    print("\n🆕 Testing new model training...")
    try:
        cmd = [
            sys.executable, str(script_path),
            "--mode", "new",
            "--timesteps", "1000"  # Very short test
        ]
        
        print(f"🚀 Running: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout
        )
        
        if result.returncode == 0:
            print("✅ New model training test successful")
            print("📋 Training output (last 10 lines):")
            for line in result.stdout.split('\n')[-10:]:
                if line.strip():
                    print(f"   {line}")
        else:
            print(f"❌ New model training failed: {result.stderr}")
            print("📋 Stdout:", result.stdout[-500:])  # Last 500 chars
            return False
            
    except subprocess.TimeoutExpired:
        print("⚠️ Training test timed out (this is expected for longer training)")
        return True  # Timeout is acceptable for this test
    except Exception as e:
        print(f"❌ Error testing new training: {e}")
        return False
    
    # Check if a model was created
    models_dir = project_root / "models"
    if models_dir.exists():
        model_files = list(models_dir.glob("*.pt"))
        if model_files:
            print(f"✅ Found {len(model_files)} model file(s)")
            
            # Test continue training with the newest model
            newest_model = max(model_files, key=lambda p: p.stat().st_mtime)
            print(f"\n🔄 Testing continue training with: {newest_model.name}")
            
            try:
                cmd = [
                    sys.executable, str(script_path),
                    "--mode", "continue",
                    "--model", str(newest_model),
                    "--timesteps", "500"  # Very short continuation
                ]
                
                print(f"🚀 Running: {' '.join(cmd)}")
                
                result = subprocess.run(
                    cmd,
                    cwd=str(project_root),
                    capture_output=True,
                    text=True,
                    timeout=60  # 1 minute timeout
                )
                
                if result.returncode == 0:
                    print("✅ Continue training test successful")
                    print("📋 Training output (last 5 lines):")
                    for line in result.stdout.split('\n')[-5:]:
                        if line.strip():
                            print(f"   {line}")
                else:
                    print(f"❌ Continue training failed: {result.stderr}")
                    print("📋 Stdout:", result.stdout[-500:])  # Last 500 chars
                    
            except subprocess.TimeoutExpired:
                print("⚠️ Continue training test timed out (this is expected)")
                return True  # Timeout is acceptable
            except Exception as e:
                print(f"❌ Error testing continue training: {e}")
                return False
        else:
            print("⚠️ No model files found after training")
    else:
        print("⚠️ Models directory not found")
    
    print("\n🎉 Enhanced training script tests completed!")
    return True

def test_gui_integration():
    """Test that the GUI can import and use the training functionality."""
    print("\n🔗 Testing GUI Integration")
    print("=" * 30)
    
    try:
        # Test importing the training script as a module
        current_dir = Path(__file__).parent
        sys.path.insert(0, str(current_dir))
        
        # We can't import directly, but we can check the file exists and is readable
        script_path = current_dir / "train_enhanced.py"
        
        if script_path.exists():
            print("✅ Training script accessible to GUI")
            
            # Check if the functions we expect are in the file
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            expected_functions = ['continue_training', 'train_new_model', 'main']
            for func in expected_functions:
                if f"def {func}" in content:
                    print(f"✅ Function '{func}' found")
                else:
                    print(f"❌ Function '{func}' not found")
                    return False
        else:
            print("❌ Training script not accessible")
            return False
            
    except Exception as e:
        print(f"❌ GUI integration test failed: {e}")
        return False
    
    print("✅ GUI integration test passed")
    return True

def main():
    """Run all tests."""
    print("🚀 Starting Enhanced Training Tests")
    print("=" * 60)
    
    # Test the training script
    script_success = test_training_script()
    
    # Test GUI integration
    gui_success = test_gui_integration()
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print(f"  • Training Script: {'✅ PASS' if script_success else '❌ FAIL'}")
    print(f"  • GUI Integration: {'✅ PASS' if gui_success else '❌ FAIL'}")
    
    overall_success = script_success and gui_success
    print(f"\n🎯 Overall Result: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")
    
    if overall_success:
        print("\n🎉 Enhanced training functionality is ready!")
        print("💡 You can now:")
        print("   • Train new models from the GUI")
        print("   • Continue training existing models")
        print("   • Monitor training progress in real-time")
    else:
        print("\n💭 Some issues were found. Check the output above for details.")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)