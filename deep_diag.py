import os
import sys
import subprocess
import platform

def diag():
    print("="*60)
    print("       \U0001F50D DEEP ENVIRONMENT DIAGNOSTIC")
    print("="*60)
    
    print(f"\U0001F5A5 OS: {platform.system()} {platform.release()} ({platform.architecture()[0]})")
    print(f"\U0001F40D Python Path: {sys.executable}")
    print(f"\U0001F40D Python Version: {sys.version}")
    
    print("\n--- \U0001F4E6 Package Check ---")
    packages = ['tensorflow', 'tensorflow-intel', 'keras', 'tf-keras', 'torch']
    for p in packages:
        try:
            res = subprocess.run(['pip', 'show', p], capture_output=True, text=True)
            if res.returncode == 0:
                for line in res.stdout.split('\n'):
                    if line.startswith('Version:') or line.startswith('Location:'):
                        print(f"{p}: {line}")
            else:
                print(f"{p}: NOT INSTALLED")
        except:
            print(f"{p}: Error checking package")

    print("\n--- \U0001F6E0 DLL Search (libiomp5md.dll) ---")
    # This DLL is the common cause of the conflict
    found_dlls = []
    site_packages = [p for p in sys.path if 'site-packages' in p]
    if site_packages:
        sp = site_packages[0]
        print(f"Searching in: {sp}")
        for root, dirs, files in os.walk(sp):
            if 'libiomp5md.dll' in files:
                full_path = os.path.join(root, 'libiomp5md.dll')
                found_dlls.append(full_path)
                print(f"⚠️ Found: {full_path}")
    
    if len(found_dlls) > 1:
        print("\n\U0001F6A8 WARNING: Multiple 'libiomp5md.dll' files found! This is likely the cause of the DLL Error.")

    print("\n--- \U0001F9EA Import Test ---")
    print("Attempting to import TensorFlow...")
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    try:
        import tensorflow as tf
        print(f"✅ TF Import Success! Version: {tf.__version__}")
        print(f"Has Keras: {hasattr(tf, 'keras')}")
        if hasattr(tf, 'keras'):
            print(f"Keras Version (via TF): {tf.keras.__version__}")
    except Exception as e:
        print(f"❌ TF Import Failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    print("\nAttempting to import Keras standalone...")
    try:
        import keras
        print(f"✅ Keras Import Success! Version: {keras.__version__}")
    except Exception as e:
        print(f"❌ Keras Import Failed: {type(e).__name__}: {e}")

    print("\n" + "="*60)

if __name__ == "__main__":
    diag()
