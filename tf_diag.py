import os
import sys

def test_diag():
    print("--- TensorFlow Diagnostic ---")
    print(f"Python version: {sys.version}")
    
    # Try workaround 1
    print("\nAttempting Workaround 1: KMP_DUPLICATE_LIB_OK=True")
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    
    try:
        import tensorflow as tf
        print(f"✅ SUCCESS: TensorFlow {tf.__version__} loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ FAILED: {type(e).__name__}: {e}")
        
    print("\nAttempting Workaround 2: Pre-loading native DLLs via ctypes (if applicable)...")
    # Sometimes manually loading zlib or msvcp helps, but we don't know the exact path.
    
    print("\nChecklist for manual fix:")
    print("1. Install MSVC++ Redistributable: https://aka.ms/vs/17/release/vc_redist.x64.exe")
    print("2. Downgrade TensorFlow: pip install tensorflow==2.18.0")
    print("3. Check for conflicting zlib DLLs in C:\\Windows\\System32 or other PATH folders.")

if __name__ == "__main__":
    test_diag()
