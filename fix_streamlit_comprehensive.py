#!/usr/bin/env python3
"""
Comprehensive script to replace use_container_width with width parameter for Streamlit compatibility.
"""

import re

def fix_streamlit_parameters():
    """Replace deprecated use_container_width with new width parameter."""
    
    file_path = "gui/app.py"
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("Original file size:", len(content))
    
    # Count occurrences before replacement
    true_count = len(re.findall(r'use_container_width\s*=\s*True', content))
    false_count = len(re.findall(r'use_container_width\s*=\s*False', content))
    
    print(f"Found {true_count} instances of use_container_width=True")
    print(f"Found {false_count} instances of use_container_width=False")
    
    # Replace use_container_width=True with width="stretch" (handle various spacing/comma patterns)
    content = re.sub(r'use_container_width\s*=\s*True', 'width="stretch"', content)
    
    # Replace use_container_width=False with width="content" (if any exist)
    content = re.sub(r'use_container_width\s*=\s*False', 'width="content"', content)
    
    # Verify replacements
    remaining_true = len(re.findall(r'use_container_width\s*=\s*True', content))
    remaining_false = len(re.findall(r'use_container_width\s*=\s*False', content))
    
    print(f"After replacement: {remaining_true} instances of use_container_width=True remain")
    print(f"After replacement: {remaining_false} instances of use_container_width=False remain")
    
    # Write the file back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Fixed Streamlit use_container_width deprecation warnings")
    print(f"   Replaced {true_count - remaining_true} instances of use_container_width=True with width='stretch'")
    print(f"   Replaced {false_count - remaining_false} instances of use_container_width=False with width='content'")

if __name__ == "__main__":
    fix_streamlit_parameters()