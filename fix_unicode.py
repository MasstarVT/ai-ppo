#!/usr/bin/env python3
"""
Script to remove problematic Unicode characters from train_enhanced.py
"""

import re

def fix_unicode_chars():
    """Replace problematic Unicode characters with ASCII equivalents."""
    
    file_path = "train_enhanced.py"
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Dictionary of emoji/Unicode to ASCII replacements
    replacements = {
        '🔄': '>>>',
        '✅': '[OK]',
        '❌': '[ERROR]',
        '📊': '[INFO]',
        '📥': '[LOAD]',
        '📈': '[DATA]',
        '🏗️': '[SETUP]',
        '🚀': '[START]',
        '💾': '[SAVE]',
        '⚡': '[FAST]',
        '🎯': '[TARGET]',
        '📋': '[LOG]',
        '🔍': '[SEARCH]',
        '⏹️': '[STOP]',
        '⏸️': '[PAUSE]',
        '🆕': '[NEW]',
        '⚠️': '[WARNING]',
        '💡': '[INFO]',
        '🛑': '[STOP]',
        '�': '***',  # Replacement character
        '  •': '  -',  # Bullet point
        '  -': '  -',  # Already safe
    }
    
    # Apply replacements
    for unicode_char, ascii_replacement in replacements.items():
        content = content.replace(unicode_char, ascii_replacement)
    
    # Write the file back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Fixed Unicode characters in train_enhanced.py")
    print("   Replaced emojis and special characters with ASCII equivalents")

if __name__ == "__main__":
    fix_unicode_chars()