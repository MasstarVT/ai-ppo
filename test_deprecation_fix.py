"""
Test script to verify the deprecation warning fix.
"""

import pandas as pd
import warnings
import sys
import os

# Capture warnings
warnings.filterwarnings('error', category=FutureWarning)

def test_pandas_styling():
    """Test that the pandas styling doesn't produce deprecation warnings."""
    print("Testing pandas styling without deprecation warnings...")
    
    try:
        # Create sample data similar to what the GUI uses
        data = {
            'Symbol': ['AAPL', 'MSFT', 'GOOGL'],
            'Price': [150.25, 280.50, 2750.80],
            'AI Signal': ['BUY', 'SELL', 'HOLD']
        }
        
        df = pd.DataFrame(data)
        
        # Define styling function (same as in GUI)
        def color_signal(val):
            if val == 'BUY':
                return 'color: green; font-weight: bold'
            elif val == 'SELL':
                return 'color: red; font-weight: bold'
            else:
                return 'color: gray'
        
        # Test the new .map() method (fixed version)
        styled_df = df.style.map(color_signal, subset=['AI Signal'])
        
        print("✅ New .map() method works without warnings!")
        
        # Test that the old .applymap() would produce warnings
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                old_styled_df = df.style.applymap(color_signal, subset=['AI Signal'])
                if w and any("applymap" in str(warning.message) for warning in w):
                    print("⚠️  Old .applymap() method produces deprecation warning (as expected)")
                else:
                    print("ℹ️  Old .applymap() method didn't produce warning in this pandas version")
        except Exception as e:
            print(f"Old method test: {e}")
        
        return True
        
    except FutureWarning as e:
        print(f"❌ FutureWarning detected: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Main test function."""
    print("🔍 Testing Pandas Deprecation Fix")
    print("=" * 40)
    
    print(f"Python version: {sys.version}")
    print(f"Pandas version: {pd.__version__}")
    print()
    
    success = test_pandas_styling()
    
    print()
    print("=" * 40)
    if success:
        print("✅ All tests passed! No deprecation warnings.")
        print("✅ GUI should now run without warnings.")
    else:
        print("❌ Tests failed. Please check the implementation.")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())