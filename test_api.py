import io
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

from fastapi.testclient import TestClient
from PIL import Image
import numpy as np

# Import app - expecting it to be in app.py
try:
    from app import app
except Exception as e:
    print(f"Failed to import app: {e}")
    # Print full traceback
    import traceback
    traceback.print_exc()
    sys.exit(1)

client = TestClient(app)

def test_remove_bg():
    # Create a dummy image
    img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    
    files = {"file": ("test.png", buf, "image/png")}
    
    print("Sending request...")
    try:
        # Increase timeout for the request
        response = client.post("/remove-bg", files=files, timeout=600.0)
        
        print(f"Status Code: {response.status_code}")
        if response.status_code != 200:
            print("Response:", response.text) 
        else:
            print("Success! Response length:", len(response.content))
    except Exception as e:
        print(f"Request failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_remove_bg()
