#!/usr/bin/env python3
"""
Script to replace use_container_width with width parameter for Streamlit compatibility.
"""

import re

def fix_streamlit_parameters():
    """Replace deprecated use_container_width with new width parameter."""
    
    file_path = "gui/app.py"
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace use_container_width=True with width="stretch" (handle various spacing/comma patterns)
    content = re.sub(r'use_container_width\s*=\s*True', 'width="stretch"', content)
    
    # Replace use_container_width=False with width="content" (if any exist)
    content = re.sub(r'use_container_width\s*=\s*False', 'width="content"', content)
    
    # Write the file back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Fixed Streamlit use_container_width deprecation warnings")
    print("   Replaced use_container_width=True with width='stretch'")
    print("   Replaced use_container_width=False with width='content'")

if __name__ == "__main__":
    fix_streamlit_parameters()