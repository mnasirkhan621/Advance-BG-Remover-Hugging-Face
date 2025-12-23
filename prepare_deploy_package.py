import tarfile
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("packer")

def create_package():
    output_filename = "deploy_package.tar.gz"
    
    # Files/Dirs to include
    includes = [
        "app.py",
        "config.py",
        "utils.py",
        "dataset.py", # Added missing file
        "image_proc.py", # Added potentially missing file
        "requirements.txt",
        "Dockerfile",
        "models",
        # "weights", # Download on server to save SCP time
        # "weights/quantized_model.pth", # Included in 'weights'
        # "weights/model.safetensors",   # Included in 'weights'
    ]
    
    # Explicit exclusions if walking directories
    excludes = [
        "__pycache__",
        "venv",
        ".git",
        ".vscode",
        "test_api.py",
        "test_model.py",
        # "quantize_test.py", # Include this
        # "verify_quantized.py", # Include this
        "deploy_package.tar.gz"
    ]

    with tarfile.open(output_filename, "w:gz") as tar:
        for item in includes:
            if os.path.exists(item):
                logger.info(f"Adding {item}...")
                tar.add(item, filter=lambda x: None if any(e in x.name for e in excludes) else x)
            else:
                logger.warning(f"Item not found: {item}")
                
    logger.info(f"Package created: {output_filename}")

if __name__ == "__main__":
    create_package()
